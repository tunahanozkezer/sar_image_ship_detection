const { ipcRenderer } = require('electron');
const fs = require('fs');
const path = require('path');

const selectDirBtn = document.getElementById('selectDirBtn');
const selectedDirP = document.getElementById('selectedDir');
const processImagesBtn = document.getElementById('processImagesBtn');
const imageList = document.getElementById('imageList');
const resultsList = document.getElementById('resultsList');
const processedImage = document.getElementById('processedImage');

// 1) KLASÖR SEÇME
selectDirBtn.addEventListener('click', async () => {
  const directoryPath = await ipcRenderer.invoke('select-directory');
  if (directoryPath) {
    selectedDirP.textContent = `Seçilen Klasör: ${directoryPath}`;
    // Klasördeki resimleri listeleyelim
    listImages(directoryPath);
  }
});

// 2) KLASÖRDEKİ RESİMLERİ LİSTELE
function listImages(directoryPath) {
  // Node entegrasyonu açık olduğu için fs modülünü kullanabiliriz:
  const files = fs.readdirSync(directoryPath);
  
  // Listeyi temizle
  imageList.innerHTML = "";
  
  // Yalnızca resim uzantılarını listele
  const validExtensions = ['.png', '.jpg', '.jpeg', '.bmp'];
  files.forEach((file) => {
    const ext = path.extname(file).toLowerCase();
    if (validExtensions.includes(ext)) {
      const li = document.createElement('li');
      li.textContent = file;
      imageList.appendChild(li);
    }
  });
}

// 3) "Görüntü İşle" BUTONUNA TIKLANDIĞINDA
processImagesBtn.addEventListener('click', () => {
  const directoryPath = selectedDirP.textContent.replace('Seçilen Klasör: ', '').trim();
  if (!directoryPath) return alert("Önce bir klasör seçiniz.");

  // IPC ile main sürecine mesaj gönderiyoruz
  ipcRenderer.send('process-images', directoryPath);
});

// 4) PYTHON İŞLEMLERİ BİTİNCE ALINAN CEVAP
ipcRenderer.on('process-images-complete', (event, data) => {
  console.log("Process Images Complete", data);

  if (data.error) {
    console.error("Hata:", data.error);
  }

  // data.parsed.results içinde JSON verimiz olmalı
  const { parsed } = data;
  if (!parsed || !parsed.results) {
    return;
  }

  // resultsList’i temizleyip dolduralım
  resultsList.innerHTML = "";
  Object.keys(parsed.results).forEach((filename) => {
    const shipCount = parsed.results[filename]["Gemi Sayisi"];
    const li = document.createElement('li');
    li.textContent = `${filename} -> Gemi Sayısı: ${shipCount}`;

    // Tıklayınca işlenmiş resmi gösterelim
    li.addEventListener('click', () => {
      // İşlenmiş resimlerin kaydedildiği klasörü parsed.output_dir ile aldık
      const outputDir = parsed.output_dir;
      const processedImagePath = path.join(outputDir, filename);
      // <img> etiketinin kaynağını bu path’e ayarlıyoruz
      processedImage.src = processedImagePath;
    });

    resultsList.appendChild(li);
  });
});
