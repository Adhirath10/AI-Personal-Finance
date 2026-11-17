const SETTINGS_KEY = "finai-settings";

export function readSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

export function writeSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

export function clearSettings() {
  localStorage.removeItem(SETTINGS_KEY);
}

export function initialsFor(name, fallback = "PF") {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return fallback;
  return parts.slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

// Shrink an uploaded picture to a small square JPEG so it fits in browser storage.
export function resizeImage(file, size = 160) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("The image could not be read."));
    reader.onload = () => {
      const image = new Image();
      image.onerror = () => reject(new Error("The file is not a supported image."));
      image.onload = () => {
        const side = Math.min(image.width, image.height);
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        canvas
          .getContext("2d")
          .drawImage(image, (image.width - side) / 2, (image.height - side) / 2, side, side, 0, 0, size, size);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      };
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}
