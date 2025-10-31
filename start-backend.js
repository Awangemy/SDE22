const { spawn } = require('child_process');
const path = require('path');

// Tentukan path ke fail main.py FastAPI
const backendPath = path.join(__dirname, 'main.py');

// Start FastAPI
const uvicorn = spawn('python', ['-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000', '--reload'], {
  stdio: 'inherit', // supaya output muncul di terminal
  cwd: path.dirname(backendPath)
});

uvicorn.on('close', (code) => {
  console.log(`FastAPI exited with code ${code}`);
});
