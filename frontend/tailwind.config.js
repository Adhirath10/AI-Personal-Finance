/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101b35",
        canvas: "#f5f7fb",
        brand: "#5378f6",
      },
    },
  },
  plugins: [],
};
