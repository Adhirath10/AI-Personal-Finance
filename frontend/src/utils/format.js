export const currency = (value, compact = false) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
    notation: compact ? "compact" : "standard",
  }).format(Number(value) || 0);

export const number = (value) =>
  new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 0,
  }).format(Number(value) || 0);

export const shortDate = (value) => {
  if (!value) return "—";
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en-IN", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }).format(date);
};

// Formats a value already on the 0-100 scale (e.g. 6 -> "6%"). Pass fractions
// such as 0.06 through ratioPercent instead.
export const percent = (value) => {
  const numeric = Number(value || 0);
  const decimals = Number.isInteger(numeric) ? 0 : 1;
  return `${numeric.toFixed(decimals)}%`;
};

export const ratioPercent = (ratio) => percent(Number(ratio || 0) * 100);
