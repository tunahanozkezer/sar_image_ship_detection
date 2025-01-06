// main.js

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      // Preload script for secure IPC
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  win.loadFile('index.html');
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  // On Windows and Linux, close the app when all windows are closed.
  if (process.platform !== 'darwin') app.quit();
});

// ---------- IPC HANDLERS ---------- //

/**
 * Handle folder selection
 */
ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true, folderPath: null, imageNames: [] };
  }

  const folderPath = result.filePaths[0];
  const files = fs.readdirSync(folderPath);

  // Filter out valid image extensions
  const imageNames = files.filter((file) => {
    const ext = path.extname(file).toLowerCase();
    return ['.png', '.jpg', '.jpeg', '.bmp'].includes(ext);
  });

  return { canceled: false, folderPath, imageNames };
});

/**
 * Handle image processing request
 */
ipcMain.handle('process-images', async (event, folderPath) => {
  return new Promise((resolve, reject) => {
    // Spawn a Python process to run `script.py`
    const pyProcess = spawn('python3', [path.join(__dirname, 'ship_detection/script.py'), folderPath]);

    let scriptOutput = '';
    let scriptError = '';

    // Collect data from stdout
    pyProcess.stdout.on('data', (data) => {
      scriptOutput += data.toString();
    });

    // Collect data from stderr
    pyProcess.stderr.on('data', (data) => {
      scriptError += data.toString();
    });

    // On script exit
    pyProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(`Python script exited with code ${code}. Error: ${scriptError}`);
      }

      // The script prints JSON at the end in `main()`'s return statement.
      // We want to parse that JSON out of scriptOutput if possible.
      // We'll look for a JSON-like structure in the output.
      try {
        // A simple approach is to parse the entire scriptOutput directly as JSON.
        // If your script prints other logs, you may need a more refined approach.


        const correctedJsonString = scriptOutput.replace(/'/g, '"');
        const parsedData = JSON.parse(correctedJsonString);
        console.log(parsedData);
        resolve(parsedData);
        
      } catch (err) {
        // If parsing fails, we can at least return the raw text
        resolve({
          error: 'Could not parse JSON from Python output',
          rawOutput: scriptOutput,
        });
      }

    });
  });
});
