import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
let reconnectAttempts = 0;

// --- вставь это в начало твоего файла (рядом с другими константами) ---
let foveatedEnabled = true;
let skipPeripheral = false; // если true — сфера за пределом порога отбрасывается (discard)
const GAZE_NDC = new THREE.Vector2(0, 0);

// radii (в NDC)
const WS_PROTOCOL = "ws";
const DEFAULT_WS_HOST = "localhost";
const WS_HOST = DEFAULT_WS_HOST;
const WS_PORT = "8765";
const WS_URL = `${WS_PROTOCOL}://${WS_HOST}:${WS_PORT}`;
let ws = null;
const maxReconnectAttempts = 10;
const RADIUS_FOVEA = 0.15 * 2.0; // примерный масштаб, можно тонко настроить
const RADIUS_PARAFOVEA = 0.35 * 2.0;
const debugDiv = document.createElement("div");
debugDiv.style.cssText = `
  position: fixed;
  top: 10px;
  left: 10px;
  background: rgba(0,0,0,0.7);
  color: #0f0;
  padding: 10px;
  font-family: monospace;
  font-size: 12px;
  z-index: 1000;
  border-radius: 5px;
  pointer-events: none;
`;
document.body.appendChild(debugDiv);

const gazeIndicator = document.createElement("div");
gazeIndicator.style.cssText = `
  position: fixed;
  width: 20px;
  height: 20px;
  border: 2px solid #0f0;
  border-radius: 50%;
  background: rgba(0, 255, 0, 0.2);
  pointer-events: none;
  z-index: 999;
  transform: translate(-50%, -50%);
  display: none;
`;
document.body.appendChild(gazeIndicator);

// Метрики
const perfSamples = [];
const PERF_SAMPLE_COUNT = 60;
let lastFrameTime = performance.now();
let frameTime = 0;
let fps = 0;
let gazeData = {
  x: 0.5,
  y: 0.5,
  confidence: 0.0,
  connected: false,
};
setTimeout(() => {
  connectWebSocket();
}, 1000); // Wait 1 second before first connection attempt

// Store point cloud data for foveated rendering
let pointCloud = null;

// Foveated rendering configuration
const FOV_CONFIG = {
  highQualityRadius: 0.15, // 15% of screen radius for high quality
  mediumQualityRadius: 0.35, // 35% for medium quality
  lowQualityRadius: 0.6, // 60% for low quality
  minPointSize: 0.0005, // Minimum point size (peripheral)
  maxPointSize: 0.002, // Maximum point size (foveal)
  minOpacity: 0.3, // Minimum opacity (peripheral)
  maxOpacity: 1.0, // Maximum opacity (foveal)
};

// --- Шейдеры для точечного облака (фовеа) ---
const vertShader = /* glsl */ `
  precision highp float;
  attribute vec3 position;
  attribute vec3 color;
  varying vec3 vColor;
  varying float vPointSizeFactor;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform vec2 gazeNDC; // gaze in NDC (-1..1)
  uniform float minPointSize;
  uniform float maxPointSize;
  uniform float minOpacity;
  uniform float maxOpacity;
  uniform float radiusFovea;
  uniform float radiusParafovea;
  uniform bool foveatedEnabled;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vec4 clipPos = projectionMatrix * mvPosition;
    // NDC coordinates
    vec2 ndc = clipPos.xy / clipPos.w;

    // if point is behind camera, push it off-screen (still processed)
    if (clipPos.w <= 0.0) {
      vPointSizeFactor = 0.0;
      gl_Position = clipPos;
      return;
    }

    // distance in NDC space
    float d = distance(ndc, gazeNDC);

    // compute smooth factor: 0 -> fovea, 1 -> peripheral
    float smoothFactor = 0.0;
    if (foveatedEnabled) {
      if (d <= radiusFovea) {
        smoothFactor = 0.0;
      } else if (d <= radiusParafovea) {
        smoothFactor = (d - radiusFovea) / (radiusParafovea - radiusFovea);
      } else {
        smoothFactor = 1.0;
      }
    } else {
      smoothFactor = 0.0;
    }

    // invert: 0 => fovea (big), 1 => peripheral (small)
    float inv = 1.0 - smoothFactor;

    // point size interpolation
    float pointSize = mix(minPointSize, maxPointSize, inv);

    // also scale by perspective (so points closer are larger)
    float perspectiveScale = clamp( (1.0 / abs(mvPosition.z)) * 40.0, 0.5, 2.0);
    pointSize *= perspectiveScale;

    vPointSizeFactor = inv; // pass for fragment alpha

    gl_Position = clipPos;
    gl_PointSize = pointSize *  (300.0 / (projectionMatrix[0][0])); // normalize size by fov
  }
`;

const fragShader = /* glsl */ `
  precision highp float;
  varying vec3 vColor;
  varying float vPointSizeFactor;
  uniform float minOpacity;
  uniform float maxOpacity;
  uniform bool foveatedEnabled;
  uniform vec2 gazeNDC;
  uniform float radiusParafovea;
  uniform bool skipPeripheral;

  void main() {
    // circular point shape
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    float alpha = mix(minOpacity, maxOpacity, vPointSizeFactor);

    // If skipPeripheral: when vPointSizeFactor is near 0 (peripheral), discard
    if (skipPeripheral && foveatedEnabled) {
      if (vPointSizeFactor < 0.15) discard; // threshold: tune as needed
    }

    gl_FragColor = vec4(vColor, alpha);
  }
`;

// --- Функция создания Shader PointsMaterial и замены pointCloud ---
function createFoveatedPointsFromGeometry(geometry) {
  // Ensure color attribute exists
  if (!geometry.attributes.color) {
    const count = geometry.attributes.position.count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      colors[i * 3 + 0] = 1.0;
      colors[i * 3 + 1] = 1.0;
      colors[i * 3 + 2] = 1.0;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }

  const shaderMat = new THREE.ShaderMaterial({
    vertexShader: vertShader,
    fragmentShader: fragShader,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    uniforms: {
      gazeNDC: { value: new THREE.Vector2(0.0, 0.0) },
      minPointSize: { value: FOV_CONFIG.minPointSize * window.innerWidth }, // scale to pixels
      maxPointSize: { value: FOV_CONFIG.maxPointSize * window.innerWidth },
      minOpacity: { value: FOV_CONFIG.minOpacity },
      maxOpacity: { value: FOV_CONFIG.maxOpacity },
      radiusFovea: { value: RADIUS_FOVEA },
      radiusParafovea: { value: RADIUS_PARAFOVEA },
      foveatedEnabled: { value: foveatedEnabled },
      skipPeripheral: { value: skipPeripheral },
    },
  });

  const points = new THREE.Points(geometry, shaderMat);
  return points;
}
const loader = new GLTFLoader();

// --- Модифицируем загрузчик PLY: ---
loader.load("/scenes/scene.ply", (geometry) => {
  geometry.computeVertexNormals();
  originalGeometry = geometry;

  // Create foveated points
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
  }

  pointCloud = createFoveatedPointsFromGeometry(geometry);

  // Center the cloud like раньше
  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  pointCloud.position.sub(center);

  scene.add(pointCloud);
});

// --- Функция обновления gaze uniform ---
function updateGazeUniform() {
  // convert gazeData (0..1) to NDC (-1..1)
  const gx = gazeData.x * 2.0 - 1.0;
  const gy = -(gazeData.y * 2.0 - 1.0); // invert y for NDC

  GAZE_NDC.set(gx, gy);

  if (pointCloud && pointCloud.material && pointCloud.material.uniforms) {
    pointCloud.material.uniforms.gazeNDC.value.set(gx, gy);
    pointCloud.material.uniforms.foveatedEnabled.value = foveatedEnabled;
    pointCloud.material.uniforms.skipPeripheral.value = skipPeripheral;
  }
}

// --- Метрики: измерение времени рендера и сбор статистики ---
function updatePerformanceMeasures(renderStart, renderEnd) {
  const ms = renderEnd - renderStart;
  frameTime = ms;
  perfSamples.push(ms);
  if (perfSamples.length > PERF_SAMPLE_COUNT) perfSamples.shift();

  const sum = perfSamples.reduce((a, b) => a + b, 0);
  const avg = sum / perfSamples.length;
  fps = 1000 / (avg || 16.67);
}

// --- Обновим debugDiv чтобы показывать метрики (замени существующую функцию updateDebugInfo) ---
function updateDebugInfo() {
  const connected = gazeData.connected;
  const conf = gazeData.confidence;
  const x = gazeData.x;
  const y = gazeData.y;

  let statusText = connected ? "✓ Connected" : "✗ Disconnected";
  if (!connected && reconnectAttempts > 0) {
    statusText += ` (retrying...)`;
  }

  // renderer.info
  const ri = renderer.info;
  const calls = ri.render.calls;
  const triangles = ri.render.triangles;
  const pointsDrawn = ri.render.points;

  const memGeom = ri.memory.geometries;
  const memTex = ri.memory.textures;

  const avgFrame = perfSamples.length
    ? (perfSamples.reduce((a, b) => a + b, 0) / perfSamples.length).toFixed(2)
    : "n/a";

  debugDiv.innerHTML = `
    <div style="color: ${connected ? "#0f0" : "#f00"}">Gaze: ${statusText}</div>
    <div>WS URL: ${WS_URL}</div>
    <div>Position: (${x.toFixed(3)}, ${y.toFixed(3)})</div>
    <div>Confidence: ${(conf * 100).toFixed(1)}%</div>
    <div>Foveated: ${
      foveatedEnabled ? "ON" : "OFF"
    } / skipPeripheral: ${skipPeripheral}</div>
    <div style="margin-top:6px">Frame: ${frameTime.toFixed(
      2
    )} ms (avg ${avgFrame} ms) | FPS ~ ${Math.round(fps)}</div>
    <div>Render calls: ${calls} | Triangles: ${triangles} | Points: ${pointsDrawn}</div>
    <div>Geometries: ${memGeom} | Textures: ${memTex}</div>
    <div style="font-size:10px; color:#ff0; margin-top:6px">Keys: F - toggle fovea, S - skip peripheral, T - log metrics</div>
  `;

  // Update gaze indicator position
  if (connected && conf > 0.2) {
    gazeIndicator.style.display = "block";
    gazeIndicator.style.left = `${x * window.innerWidth}px`;
    gazeIndicator.style.top = `${y * window.innerHeight}px`;
  } else {
    gazeIndicator.style.display = "none";
  }
}

// --- Обновлённый animate() с метриками ---
// ------------------ GLOBAL INIT (создаётся ОДИН раз) ------------------

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ------------------ ANIMATE ------------------

function animate() {
  requestAnimationFrame(animate);

  // обновляем gaze для шейдера
  updateGazeUniform();

  controls.update();

  const t0 = performance.now();
  renderer.render(scene, camera);
  const t1 = performance.now();

  updatePerformanceMeasures(t0, t1);
  updateDebugInfo();
}

animate();

// --- Клавиши для тестирования ---
window.addEventListener("keydown", (e) => {
  if (e.key === "f" || e.key === "F") {
    foveatedEnabled = !foveatedEnabled;
    console.log("[F] foveatedEnabled:", foveatedEnabled);
  } else if (e.key === "s" || e.key === "S") {
    skipPeripheral = !skipPeripheral;
    console.log("[S] skipPeripheral:", skipPeripheral);
  } else if (e.key === "t" || e.key === "T") {
    console.table({
      frameTime,
      avgFrame: perfSamples.length
        ? (perfSamples.reduce((a, b) => a + b, 0) / perfSamples.length).toFixed(
            2
          )
        : "n/a",
      fps: Math.round(fps),
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      points: renderer.info.render.points,
    });
  }
});

function connectWebSocket() {
  try {
    console.log(`Attempting to connect to ${WS_URL}...`);
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("✓ Connected to gaze tracking server");
      gazeData.connected = true;
      reconnectAttempts = 0; // Reset on successful connection
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        gazeData.x = data.x;
        gazeData.y = data.y;
        gazeData.confidence = data.confidence || 0.0;
        gazeData.distance = data.distance || 0.6;
      } catch (e) {
        console.error("Error parsing gaze data:", e);
      }
    };

    ws.onerror = (error) => {
      console.warn("WebSocket error:", error);
      gazeData.connected = false;
      // Don't log connection refused errors repeatedly
      if (reconnectAttempts === 0 || reconnectAttempts % 5 === 0) {
        console.log(
          `Make sure the Python eye-tracking module is running and reachable at ${WS_URL}!`
        );
      }
    };

    ws.onclose = (event) => {
      console.log(
        "WebSocket closed. Code:",
        event.code,
        "Reason:",
        event.reason,
        "| URL:",
        WS_URL
      );
      gazeData.connected = false;

      // Only reconnect if it wasn't a manual close
      if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        const delay = Math.min(2000 * reconnectAttempts, 10000); // Max 10 seconds
        console.log(
          `Reconnecting to ${WS_URL} in ${
            delay / 1000
          }s... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`
        );
        setTimeout(connectWebSocket, delay);
      } else if (reconnectAttempts >= maxReconnectAttempts) {
        console.error(
          "Max reconnection attempts reached. Please restart the Python eye-tracking module."
        );
      }
    };
  } catch (error) {
    console.error("Failed to create WebSocket connection:", error, WS_URL);
    gazeData.connected = false;
    if (reconnectAttempts < maxReconnectAttempts) {
      reconnectAttempts++;
      setTimeout(connectWebSocket, 2000);
    }
  }
}

connectWebSocket();
