import * as THREE from "three";
import { PLYLoader } from "three-stdlib";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  5000
);
camera.position.set(1, 1, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

// Foveated rendering parameters
let gazeData = {
  x: 0.5,
  y: 0.5,
  confidence: 0.0,
  connected: false
};

// Foveated rendering configuration
const FOV_CONFIG = {
  highQualityRadius: 0.15,      // 15% of screen radius for high quality
  mediumQualityRadius: 0.35,    // 35% for medium quality
  lowQualityRadius: 0.6,        // 60% for low quality
  minPointSize: 0.0005,         // Minimum point size (peripheral)
  maxPointSize: 0.002,          // Maximum point size (foveal)
  minOpacity: 0.3,              // Minimum opacity (peripheral)
  maxOpacity: 1.0               // Maximum opacity (foveal)
};

const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const DEFAULT_WS_HOST = window.location.hostname || "localhost";
const WS_HOST =
  import.meta.env.VITE_GAZE_WS_HOST?.trim() || DEFAULT_WS_HOST;
const WS_PORT = import.meta.env.VITE_GAZE_WS_PORT?.trim() || "8765";
const WS_URL =
  import.meta.env.VITE_GAZE_WS_URL?.trim() ||
  `${WS_PROTOCOL}://${WS_HOST}:${WS_PORT}`;

// WebSocket connection for gaze data
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;

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
          `Reconnecting to ${WS_URL} in ${delay / 1000}s... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`
        );
        setTimeout(connectWebSocket, delay);
      } else if (reconnectAttempts >= maxReconnectAttempts) {
        console.error("Max reconnection attempts reached. Please restart the Python eye-tracking module.");
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

// Connect to gaze tracking server (wait a bit for Python server to start)
setTimeout(() => {
  connectWebSocket();
}, 1000); // Wait 1 second before first connection attempt

// Store point cloud data for foveated rendering
let pointCloud = null;
let originalGeometry = null;

// Load .ply
const loader = new PLYLoader();
loader.load("/scenes/scene.ply", (geometry) => {
  geometry.computeVertexNormals();
  originalGeometry = geometry;

  const material = new THREE.PointsMaterial({
    size: FOV_CONFIG.maxPointSize,
    vertexColors: true,
    transparent: true,
    opacity: FOV_CONFIG.maxOpacity
  });

  pointCloud = new THREE.Points(geometry, material);

  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  pointCloud.position.sub(center);

  scene.add(pointCloud);
});

// Foveated rendering using multiple point clouds with different LODs
let fovealPoints = null;
let parafovealPoints = null;
let peripheralPoints = null;

function createFoveatedPointClouds(geometry) {
  const positions = geometry.attributes.position;
  const colors = geometry.attributes.color;
  const count = positions.count;

  // Create separate geometries for different quality regions
  const fovealIndices = [];
  const parafovealIndices = [];
  const peripheralIndices = [];

  // We'll dynamically update which points belong to which region based on gaze
  // For now, create the point clouds
  const fovealGeo = geometry.clone();
  const parafovealGeo = geometry.clone();
  const peripheralGeo = geometry.clone();

  // Create materials with different qualities
  const fovealMaterial = new THREE.PointsMaterial({
    size: FOV_CONFIG.maxPointSize,
    vertexColors: true,
    transparent: true,
    opacity: FOV_CONFIG.maxOpacity
  });

  const parafovealMaterial = new THREE.PointsMaterial({
    size: FOV_CONFIG.maxPointSize * 0.6,
    vertexColors: true,
    transparent: true,
    opacity: FOV_CONFIG.maxOpacity * 0.7
  });

  const peripheralMaterial = new THREE.PointsMaterial({
    size: FOV_CONFIG.minPointSize,
    vertexColors: true,
    transparent: true,
    opacity: FOV_CONFIG.minOpacity
  });

  fovealPoints = new THREE.Points(fovealGeo, fovealMaterial);
  parafovealPoints = new THREE.Points(parafovealGeo, parafovealMaterial);
  peripheralPoints = new THREE.Points(peripheralGeo, peripheralMaterial);

  // Initially hide all but one (we'll show/hide based on gaze)
  parafovealPoints.visible = false;
  peripheralPoints.visible = false;

  return { fovealPoints, parafovealPoints, peripheralPoints };
}

// Simplified foveated rendering - adjust point size and opacity based on gaze
function applyFoveatedRendering() {
  if (!pointCloud || gazeData.confidence < 0.3) {
    // Fallback to uniform rendering if gaze data is unreliable
    if (pointCloud) {
      pointCloud.material.size = FOV_CONFIG.maxPointSize;
      pointCloud.material.opacity = FOV_CONFIG.maxOpacity;
    }
    return;
  }

  // Calculate distance from center of screen (where user is looking)
  // Gaze coordinates are normalized (0-1), convert to screen space (-1 to 1)
  const centerX = 0; // Center of screen in normalized coords
  const centerY = 0;
  const gazeX = (gazeData.x - 0.5) * 2;
  const gazeY = (0.5 - gazeData.y) * 2; // Invert Y
  
  // For now, use a gradient effect: higher quality at gaze point
  // This is a simplified version - full implementation would require per-point shaders
  const distanceFromCenter = Math.sqrt(gazeX * gazeX + gazeY * gazeY);
  
  // Adjust overall quality based on gaze position
  // When looking at center, use high quality; when looking away, reduce quality
  const qualityFactor = Math.max(0.3, 1.0 - distanceFromCenter * 0.5);
  
  pointCloud.material.size = THREE.MathUtils.lerp(
    FOV_CONFIG.minPointSize,
    FOV_CONFIG.maxPointSize,
    qualityFactor
  );
  pointCloud.material.opacity = THREE.MathUtils.lerp(
    FOV_CONFIG.minOpacity,
    FOV_CONFIG.maxOpacity,
    qualityFactor
  );
}

// Debug visualization with gaze point indicator
const debugDiv = document.createElement('div');
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

// Gaze point indicator
const gazeIndicator = document.createElement('div');
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

function updateDebugInfo() {
  const connected = gazeData.connected;
  const conf = gazeData.confidence;
  const x = gazeData.x;
  const y = gazeData.y;
  
  let statusText = connected ? '✓ Connected' : '✗ Disconnected';
  if (!connected && reconnectAttempts > 0) {
    statusText += ` (retrying...)`;
  }
  
  debugDiv.innerHTML = `
    <div style="color: ${connected ? '#0f0' : '#f00'}">Gaze: ${statusText}</div>
    <div>WS URL: ${WS_URL}</div>
    <div>Position: (${x.toFixed(3)}, ${y.toFixed(3)})</div>
    <div>Confidence: ${(conf * 100).toFixed(1)}%</div>
    <div>Foveated: ${conf > 0.3 ? 'ON' : 'OFF'}</div>
    ${!connected ? '<div style="font-size: 10px; color: #ff0; margin-top: 5px;">Start Python module first!</div>' : ''}
  `;
  
  // Update gaze indicator position
  if (connected && conf > 0.3) {
    gazeIndicator.style.display = 'block';
    gazeIndicator.style.left = `${x * window.innerWidth}px`;
    gazeIndicator.style.top = `${y * window.innerHeight}px`;
  } else {
    gazeIndicator.style.display = 'none';
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  
  // Apply foveated rendering
  applyFoveatedRendering();
  
  // Update debug info
  updateDebugInfo();
  
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
