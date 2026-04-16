export function toPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A'
  }
  return `${(value * 100).toFixed(1)}%`
}

export function toFixed(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A'
  }
  return value.toFixed(digits)
}

export function titleCase(value: string): string {
  return value
    .replaceAll('_', ' ')
    .split(' ')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}
