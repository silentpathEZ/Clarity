// bridge.js - Run this with: node bridge.js
const dgram = require("dgram");
const WebSocket = require("ws");

// UDP Server Setup
const UDP_PORT = 5052;
const udpServer = dgram.createSocket("udp4");

// WebSocket Server Setup
const WS_PORT = 5053;
const wss = new WebSocket.Server({ port: WS_PORT });

let wsClients = [];
let messageCount = 0; // Counter to track received messages

// Handle WebSocket connections
wss.on("connection", (ws) => {
  console.log("✅ New WebSocket client connected");
  wsClients.push(ws);

  ws.on("close", () => {
    console.log("❌ WebSocket client disconnected");
    wsClients = wsClients.filter((client) => client !== ws);
  });

  ws.on("error", (error) => {
    console.error("WebSocket error:", error);
  });
});

// Handle UDP messages
udpServer.on("message", (msg, rinfo) => {
  try {
    messageCount++;

    // Log every 30 messages to avoid spam (or log every message if debugging)
    if (messageCount % 30 === 0) {
      console.log(
        `📦 Received UDP packet #${messageCount} from ${rinfo.address}:${rinfo.port}`
      );
    }

    const jsonData = msg.toString("utf-8");
    const data = JSON.parse(jsonData);

    // Log first message in full to verify structure
    if (messageCount === 1) {
      console.log(
        "📋 First message structure:",
        JSON.stringify(data).substring(0, 200) + "..."
      );
      console.log(
        `📊 Landmarks count: ${data.landmarks ? data.landmarks.length : 0}`
      );
    }

    // Forward to all connected WebSocket clients
    let sentCount = 0;
    wsClients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(data));
        sentCount++;
      }
    });

    // Log forwarding status periodically
    if (messageCount % 30 === 0) {
      console.log(
        `📡 Forwarded to ${sentCount}/${wsClients.length} WebSocket client(s)`
      );
    }
  } catch (error) {
    console.error("❌ Error processing UDP message:", error.message);
    console.error("Raw message:", msg.toString("utf-8").substring(0, 100));
  }
});

udpServer.on("listening", () => {
  const address = udpServer.address();
  console.log(`\n${"=".repeat(60)}`);
  console.log(`🎧 UDP Server listening on ${address.address}:${address.port}`);
  console.log(`🌐 WebSocket Server listening on ws://localhost:${WS_PORT}`);
  console.log(`${"=".repeat(60)}`);
  console.log(`✨ Bridge is ready!`);
  console.log(
    `   1. Python script should send to UDP ${address.address}:${address.port}`
  );
  console.log(`   2. Browser should connect to ws://localhost:${WS_PORT}`);
  console.log(`${"=".repeat(60)}\n`);
});

udpServer.on("error", (err) => {
  console.error("❌ UDP Server error:", err.message);

  if (err.code === "EADDRINUSE") {
    console.error(
      `⚠️  Port ${UDP_PORT} is already in use. Close other instances or change the port.`
    );
  }

  udpServer.close();
  process.exit(1);
});

wss.on("error", (err) => {
  console.error("❌ WebSocket Server error:", err.message);

  if (err.code === "EADDRINUSE") {
    console.error(
      `⚠️  Port ${WS_PORT} is already in use. Close other instances or change the port.`
    );
  }

  process.exit(1);
});

// Start listening
udpServer.bind(UDP_PORT);

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\n🛑 Shutting down bridge server...");
  console.log(`📊 Total messages received: ${messageCount}`);
  udpServer.close();
  wss.close();
  process.exit(0);
});

// Periodic status report
setInterval(() => {
  if (messageCount > 0) {
    console.log(
      `\n📊 Status: ${messageCount} total messages | ${wsClients.length} WebSocket client(s) connected`
    );
  }
}, 10000); // Every 10 seconds