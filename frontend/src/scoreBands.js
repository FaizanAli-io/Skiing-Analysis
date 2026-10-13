// The 12-step Blue IQ scale (60–240), shared by the portal and charts.
export function scoreBand(score) {
  const value = Number(score || 0);
  if (value >= 225) return ["Mastering 3", "mastering"];
  if (value >= 210) return ["Mastering 2", "mastering"];
  if (value >= 195) return ["Mastering 1", "mastering"];
  if (value >= 180) return ["Advancing 3", "advancing"];
  if (value >= 165) return ["Advancing 2", "advancing"];
  if (value >= 150) return ["Advancing 1", "advancing"];
  if (value >= 135) return ["Progressing 3", "progressing"];
  if (value >= 120) return ["Progressing 2", "progressing"];
  if (value >= 105) return ["Progressing 1", "progressing"];
  if (value >= 90) return ["Building 3", "building"];
  if (value >= 75) return ["Building 2", "building"];
  return ["Building 1", "building"];
}
