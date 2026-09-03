import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { FingerScript } from "./library/FingerScript";
import { HandController } from "./library/HandController";
import { FingerHingeCon } from "./library/FingerHingeCon";
import { FingerHingeThumbCon } from "./library/FingerHingeThumb";
import { fingerThumbmid } from "./library/fingerThumbmid";

function updateStatus(message, type) {
  console.log(`[${type.toUpperCase()}] ${message}`);
  const statusEl = document.getElementById("status");
  if (statusEl) {
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
  }
}

class HandTrackingApp {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);

    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.01,
      1000
    );
    this.camera.position.set(0, 20, 20);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);

    this.debugMode = true;
    this.testMode = false;
    this.frameCount = 0;
    this.lastDataUpdate = 0;
    this.messageCount = 0;
    this.ws = null;
    this.currentGesture = "";
    this.gestureTimer = null;

    // Smooth movement system
    this.smoothFactor = 0.35;
    this.targetPositions = [];
    this.smoothInitialized = false;

    this.createGestureDisplay();
    this.createDebugUI();
    this.setupLights();
    this.setupControls();
    this.createLandmarkSpheres();
    this.loadHandModel();

    window.addEventListener("resize", () => this.onWindowResize(), false);

    this.animate();
    this.setupWebSocket();
  }

  createGestureDisplay() {
    const div = document.createElement("div");
    div.id = "gesture-display";
    div.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.8);
      color: #00ff88;
      padding: 20px 40px;
      border-radius: 15px;
      font-family: 'Arial', sans-serif;
      font-size: 48px;
      font-weight: bold;
      text-align: center;
      z-index: 1000;
      border: 2px solid #00ff88;
      box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
      opacity: 0;
      transition: opacity 0.3s ease;
      pointer-events: none;
    `;
    div.textContent = "";
    document.body.appendChild(div);
    this.gestureDisplay = div;
  }

  updateGestureDisplay() {
    if (!this.gestureDisplay) return;
    if (this.currentGesture) {
      this.gestureDisplay.textContent = this.currentGesture;
      this.gestureDisplay.style.opacity = "1";
    } else {
      this.gestureDisplay.style.opacity = "0";
    }
  }

  createDebugUI() {
    const debugContainer = document.createElement("div");
    debugContainer.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.8);
      color: #0f0;
      padding: 15px;
      font-family: monospace;
      font-size: 12px;
      border-radius: 5px;
      z-index: 1000;
      max-width: 300px;
    `;
    debugContainer.innerHTML = `
      <div id="status" style="margin-bottom: 10px; color: #fff;">Initializing...</div>
      <div id="debug-info"></div>
      <div style="margin-top: 10px;">
        <label style="color: #aaa; font-size: 11px;">Smoothness:</label>
        <input type="range" id="smooth-slider" min="0" max="100" value="35"
               style="width: 100%; margin-top: 3px;">
        <div style="color: #888; font-size: 10px;" id="smooth-val">35%</div>
      </div>
      <button id="toggle-test" style="margin-top: 10px; padding: 5px 10px; cursor: pointer;">
        Enable Test Mode
      </button>
      <button id="toggle-spheres" style="margin-top: 5px; padding: 5px 10px; cursor: pointer;">
        Toggle Spheres
      </button>
    `;
    document.body.appendChild(debugContainer);

    document.getElementById("toggle-test").addEventListener("click", () => {
      this.testMode = !this.testMode;
      document.getElementById("toggle-test").textContent = this.testMode
        ? "Disable Test Mode"
        : "Enable Test Mode";
      updateStatus(
        this.testMode
          ? "Test mode enabled - simulating hand movement"
          : "Test mode disabled",
        "info"
      );
    });

    document.getElementById("toggle-spheres").addEventListener("click", () => {
      const visible = !this.handPoints[0].visible;
      this.handPoints.forEach((p) => (p.visible = visible));
      this.landmarkLines.forEach(({ line }) => (line.visible = visible));
    });

    // Smooth slider
    const smoothSlider = document.getElementById("smooth-slider");
    const smoothVal = document.getElementById("smooth-val");
    if (smoothSlider) {
      smoothSlider.addEventListener("input", (e) => {
        this.smoothFactor = e.target.value / 100;
        smoothVal.textContent = e.target.value + "%";
      });
    }
  }

  updateDebugInfo() {
    if (!this.debugMode) return;

    const debugEl = document.getElementById("debug-info");
    if (!debugEl) return;

    const timeSinceUpdate = Date.now() - this.lastDataUpdate;
    const receiving = timeSinceUpdate < 1000;
    const wsConnected = this.ws && this.ws.readyState === WebSocket.OPEN;

    debugEl.innerHTML = `
      <div style="color: ${wsConnected ? "#0f0" : "#f00"}">
        WS: ${wsConnected ? "CONNECTED" : "DISCONNECTED"}
      </div>
      <div style="color: ${receiving ? "#0f0" : "#f00"}">
        Data: ${receiving ? "RECEIVING" : "NO DATA"}
      </div>
      <div>Messages: ${this.messageCount || 0}</div>
    `;
  }

  setupLights() {
    const d = new THREE.DirectionalLight(0xffffff, 1.2);
    d.position.set(5, 10, 7.5);
    this.scene.add(d);

    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambient);

    const hem = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
    this.scene.add(hem);
  }

  setupControls() {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1, 0);
  }

  createLandmarkSpheres() {
    this.handPoints = [];
    const geo = new THREE.SphereGeometry(0.02, 16, 16);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x00ff88,
      emissive: 0x00aa44,
      emissiveIntensity: 0.5,
      metalness: 0.3,
      roughness: 0.7,
    });

    for (let i = 0; i < 21; i++) {
      const s = new THREE.Mesh(geo, mat);
      s.visible = true;
      s.position.set(0, 1, 0);
      this.scene.add(s);
      this.handPoints.push(s);
    }

    this.createLandmarkConnections();
  }

  createLandmarkConnections() {
    const connections = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4], // Thumb
      [0, 5],
      [5, 6],
      [6, 7],
      [7, 8], // Index
      [0, 9],
      [9, 10],
      [10, 11],
      [11, 12], // Middle
      [0, 13],
      [13, 14],
      [14, 15],
      [15, 16], // Ring
      [0, 17],
      [17, 18],
      [18, 19],
      [19, 20], // Pinky
      [5, 9],
      [9, 13],
      [13, 17], // Palm
    ];

    this.landmarkLines = [];
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00ff88,
      opacity: 0.6,
      transparent: true,
    });

    connections.forEach(([s, e]) => {
      const geo = new THREE.BufferGeometry();
      const positions = new Float32Array(6);
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

      const line = new THREE.Line(geo, lineMat);
      this.scene.add(line);

      this.landmarkLines.push({ line, s, e });
    });
  }

  updateLandmarkConnections() {
    if (!this.landmarkLines) return;

    this.landmarkLines.forEach(({ line, s, e }) => {
      const positions = line.geometry.attributes.position.array;
      const p1 = this.handPoints[s].position;
      const p2 = this.handPoints[e].position;

      positions[0] = p1.x;
      positions[1] = p1.y;
      positions[2] = p1.z;
      positions[3] = p2.x;
      positions[4] = p2.y;
      positions[5] = p2.z;

      line.geometry.attributes.position.needsUpdate = true;
    });
  }

  simulateHandMovement() {
    const time = Date.now() * 0.001;
    const closeFactor = (Math.sin(time) + 1) * 0.5;

    this.handPoints[0].position.set(0, 1, 0);
    this.handPoints[1].position.set(0.05, 1, 0.05);
    this.handPoints[2].position.set(0.08, 1, 0.08);
    this.handPoints[3].position.set(0.11, 1, 0.11);
    this.handPoints[4].position.set(0.14 + closeFactor * 0.05, 1, 0.14);
    this.handPoints[5].position.set(0.1, 1, 0);
    this.handPoints[6].position.set(0.15, 1, 0);
    this.handPoints[7].position.set(0.2, 1, 0);
    this.handPoints[8].position.set(0.25 + closeFactor * 0.1, 1, 0);
    this.handPoints[9].position.set(0.1, 1, -0.05);
    this.handPoints[10].position.set(0.15, 1, -0.05);
    this.handPoints[11].position.set(0.2, 1, -0.05);
    this.handPoints[12].position.set(0.25 + closeFactor * 0.1, 1, -0.05);
    this.handPoints[13].position.set(0.08, 1, -0.1);
    this.handPoints[14].position.set(0.13, 1, -0.1);
    this.handPoints[15].position.set(0.18, 1, -0.1);
    this.handPoints[16].position.set(0.23 + closeFactor * 0.1, 1, -0.1);
    this.handPoints[17].position.set(0.05, 1, -0.15);
    this.handPoints[18].position.set(0.1, 1, -0.15);
    this.handPoints[19].position.set(0.15, 1, -0.15);
    this.handPoints[20].position.set(0.2 + closeFactor * 0.08, 1, -0.15);

    this.lastDataUpdate = Date.now();
  }

  loadHandModel() {
    const loader = new FBXLoader();
    this.bones = {};
    this.finalControllers = [];

    loader.load(
      "public/righthand.fbx",
      (fbx) => {
        this.model = fbx;
        this.model.scale.set(0.01, 0.01, 0.01);
        this.scene.add(this.model);

        // Discover bones
        fbx.traverse((child) => {
          if (child.isBone) {
            const m = child.name.match(/\d+/);
            if (m) {
              const num = parseInt(m[0], 10);
              if (!Number.isNaN(num)) this.bones[num] = child;
            }
          }
        });

        const skinned = [];
        fbx.traverse((c) => {
          if (c.isSkinnedMesh) skinned.push(c);
        });

        if (Object.keys(this.bones).length < 20 && skinned.length > 0) {
          skinned.forEach((sm) => {
            if (sm.skeleton && sm.skeleton.bones) {
              sm.skeleton.bones.forEach((b, idx) => {
                if (!this.bones[idx]) this.bones[idx] = b;
              });
            }
          });
        }

        console.log("Bones discovered:", Object.keys(this.bones).length);

        // ==== FIND ARMATURE FOR HAND ROTATION ====
        let armatureObject = null;
        fbx.traverse((child) => {
          if (
            child.type === "Object3D" &&
            child.name.toLowerCase().includes("armature")
          ) {
            armatureObject = child;
          }
          if (
            !armatureObject &&
            child.isBone &&
            child.parent &&
            !child.parent.isBone
          ) {
            armatureObject = child.parent;
          }
        });

        // CREATE HAND ROTATION CONTROLLER (entire hand orientation)
        if (armatureObject) {
          this.handConController = new HandController(
            this.handPoints[0], // Wrist
            this.handPoints[9], // Middle finger knuckle
            armatureObject,
            -1,
            -1,
            -1,
            50,
            50,
            0
          );
          console.log("✓ HandCon controller created for:", armatureObject.name);
        } else if (this.bones[0]) {
          // Fallback: use bone[0] if armature not found
          this.handConController = new HandCon(
            this.handPoints[0],
            this.handPoints[9],
            this.bones[0],
            -1,
            1,
            1,
            1,
            0,
            0
          );
          console.log("✓ HandCon controller created for bone[0] (fallback)");
        } else {
          console.warn("✗ Could not create HandCon controller");
        }

        // ==== FINGER CONTROLLERS ====
        const controllerConfigs = [
          {
            lm1: 5,
            lm2: 6,
            bone: 6,
            factorX: -1,
            factorY: -1,
            offsetX: 30,
            offsetY: 0,
          },
          {
            lm1: 6,
            lm2: 7,
            bone: 7,
            factorX: 1,
            factorY: -1,
            offsetX: -50,
            offsetY: 0,
          },
          {
            lm1: 7,
            lm2: 8,
            bone: 8,
            factorX: 0.5,
            factorY: -1,
            offsetX: -40,
            offsetY: 0,
          },
          {
            lm1: 9,
            lm2: 10,
            bone: 10,
            factorX: -1,
            factorY: 1,
            offsetX: 30,
            offsetY: 0,
          },
          {
            lm1: 13,
            lm2: 16,
            bone: 15,
            factorX: -1,
            factorY: 1,
            offsetX: 40,
            offsetY: 0,
          },
          {
            lm1: 17,
            lm2: 20,
            bone: 19,
            factorX: -1,
            factorY: 1,
            offsetX: 40,
            offsetY: 0,
          },
        ];

        controllerConfigs.forEach((config) => {
          if (this.bones[config.bone]) {
            this.finalControllers.push(
              new FingerScript(
                this.handPoints[config.lm1],
                this.handPoints[config.lm2],
                this.bones[config.bone],
                {
                  factorX: config.factorX,
                  factorY: config.factorY,
                  offsetX: config.offsetX,
                  offsetY: config.offsetY,
                }
              )
            );
          }
        });

        // Thumb controllers

        if (this.bones[2]) {
          this.finalControllers.push(
            new fingerThumbmid(
              this.handPoints[2],
              this.handPoints[3],
              this.bones[2],
              { axis: "Z", factor: 1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }
        if (this.bones[3]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[4],
              this.handPoints[3],
              this.bones[3],
              { axis: "Z", factor: 1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }

        // Middle finger hinge
        if (this.bones[12]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[10],
              this.handPoints[11],
              this.bones[12],
              { axis: "Z", factor: -1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }
        if (this.bones[13]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[11],
              this.handPoints[12],
              this.bones[13],
              { axis: "Z", factor: -1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }

        // Ring finger hinge
        if (this.bones[16]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[14],
              this.handPoints[15],
              this.bones[16],
              { axis: "Z", factor: -1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }
        if (this.bones[17]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[15],
              this.handPoints[16],
              this.bones[17],
              { axis: "Z", factor: -1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }

        // Pinky hinge
        if (this.bones[20]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[18],
              this.handPoints[19],
              this.bones[20],
              { axis: "Z", factor: -1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }
        if (this.bones[21]) {
          this.finalControllers.push(
            new FingerHingeCon(
              this.handPoints[19],
              this.handPoints[20],
              this.bones[21],
              { axis: "Z", factor: -1, minRot: 0, maxRot: 100, offset: 0 }
            )
          );
        }

        console.log(
          `✓ Created ${this.finalControllers.length} controllers total`
        );
        updateStatus("Model loaded. Ready!", "success");
      },
      (xhr) => {
        if (xhr.total) {
          const percent = Math.round((xhr.loaded / xhr.total) * 100);
          updateStatus(`Loading model: ${percent}%`, "info");
        }
      },
      (err) => {
        console.error("FBX load error", err);
        updateStatus("Failed to load hand model", "error");
      }
    );
  }

  setupWebSocket() {
    console.log("🔌 Setting up WebSocket connection...");

    try {
      this.ws = new WebSocket("ws://localhost:5053");

      this.ws.onopen = () => {
        console.log("✅ WebSocket connected!");
        updateStatus("Connected to hand tracking server", "success");
        this.lastDataUpdate = Date.now();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.messageCount = (this.messageCount || 0) + 1;

          if (this.messageCount === 1) {
            console.log("First message received:", data);
          }

          // Check for gesture name
          if (data.gesture_name) {
            this.currentGesture = data.gesture_name;
            // Clear after 3 seconds
            if (this.gestureTimer) clearTimeout(this.gestureTimer);
            this.gestureTimer = setTimeout(() => {
              this.currentGesture = "";
            }, 3000);
          }

          if (data.landmarks && Array.isArray(data.landmarks)) {
            this.updateHandLandmarks(data.landmarks);
            this.lastDataUpdate = Date.now();
          }
        } catch (error) {
          console.error("Error parsing message:", error);
        }
      };

      this.ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        updateStatus("WebSocket connection error", "error");
      };

      this.ws.onclose = () => {
        console.log("🔌 WebSocket disconnected");
        updateStatus("Disconnected from hand tracking", "warning");
        setTimeout(() => this.setupWebSocket(), 2000);
      };
    } catch (error) {
      console.error("❌ Failed to create WebSocket:", error);
      updateStatus("Failed to connect - check console", "error");
    }
  }

  updateHandLandmarks(landmarks) {
    if (landmarks.length !== 21) return;

    // Initialize positions from first frame
    if (!this.smoothInitialized || this.targetPositions.length !== 21) {
      this.targetPositions = landmarks.map((l) => ({
        x: 32.83 - l.x / 100,
        y: -l.y / 100,
        z: l.z / 100,
      }));
      this.smoothInitialized = true;
      this.handPoints.forEach((p) => (p.visible = true));
      return;
    }

    // Fixed wrist position - never move it
    const fixedWrist = this.targetPositions[0];

    // Update only finger landmarks (skip wrist = index 0)
    landmarks.forEach((landmark, index) => {
      if (index === 0) return; // Skip wrist
      if (landmark.x !== undefined) {
        this.targetPositions[index] = {
          x: 32.83 - landmark.x / 100,
          y: -landmark.y / 100,
          z: landmark.z / 100,
        };
      }
    });

    // Keep wrist locked at initial position
    this.targetPositions[0] = fixedWrist;
  }

  // Called every frame - smoothly interpolate toward targets
  smoothUpdate() {
    if (!this.smoothInitialized || this.targetPositions.length !== 21) return;

    for (let i = 0; i < 21; i++) {
      if (!this.handPoints[i]) continue;

      const target = this.targetPositions[i];
      const current = this.handPoints[i].position;

      let sf = this.smoothFactor;
      if (i === 0) sf *= 0.7;
      else if (i <= 4) sf *= 0.85;
      else if ([5, 9, 13, 17].includes(i)) sf *= 0.8;

      current.x += (target.x - current.x) * (1 - sf);
      current.y += (target.y - current.y) * (1 - sf);
      current.z += (target.z - current.z) * (1 - sf);
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.frameCount++;

    if (this.testMode) {
      this.simulateHandMovement();
    }

    // ===== SMOOTH INTERPOLATION =====
    this.smoothUpdate();

    // ===== CRITICAL: UPDATE HAND ROTATION FIRST =====
    if (this.handConController) {
      this.handConController.update();
    }

    // Then update finger controllers
    if (this.finalControllers) {
      this.finalControllers.forEach((c) => c.update());
    }

    this.updateLandmarkConnections();
    this.updateGestureDisplay();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);

    if (this.frameCount % 10 === 0) {
      this.updateDebugInfo();
    }
  }
}

new HandTrackingApp();