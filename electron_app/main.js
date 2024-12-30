const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Pencere oluşturma fonksiyonu
function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'), // varsa preload
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadFile('index.html');
}

// Uygulama hazır olduğunda pencere oluştur
app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Uygulama tüm pencereler kapandığında çık
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// IPC (Render-ana süreç iletişimi)
ipcMain.handle('select-directory', async (event) => {
  // Kullanıcıdan bir klasör seçmesini istiyoruz

  const result = await dialog.showOpenDialog({
    properties: ['openDirectory']
  });
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0]; // seçilen klasör yolu
  }
  return null;
});

ipcMain.on('process-images', (event, directoryPath) => {

  // Python scriptimizi child_process ile çalıştırıyoruz
  const scriptPath = path.join(__dirname, 'ship_detection/script.py');
  
  // Burada `python` yerine kendi sanal environment'ınıza giden path'i koyabilirsiniz.
  const pyProcess = spawn('python3', [scriptPath, directoryPath]);

  let scriptOutput = "";
  let scriptError = "";

  // Python’dan gelen stdout
  pyProcess.stdout.on('data', (data) => {
    scriptOutput += data.toString();
  });

  // Python’dan gelen stderr
  pyProcess.stderr.on('data', (data) => {
    scriptError += data.toString();
    console.error("STDERR: ", data.toString());
  });

  // İşlem bittiğinde
  pyProcess.on('close', (code) => {
    // Eğer script sonunda JSON basıyorsa, scriptOutput içinden parse edebiliriz.
    // Örneğin `json.loads()`'a karşılık Node tarafında `JSON.parse(scriptOutput)`
    let parsedData = {};
    try {
      parsedData = JSON.parse(scriptOutput);
    } catch (err) {
      console.error("JSON parse hatası:", err);
    }

    event.reply('process-images-complete', {
      code,
      output: scriptOutput,
      error: scriptError,
      parsed: parsedData
    });
  });
});
