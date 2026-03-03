import {
  formatExpectedFiles,
  normalizeExpectedFiles,
  parseMinSizeBytes,
  PROTECTED_MODEL_TYPES,
} from "../Models";

describe('Models helpers', () => {
  it('normalizeExpectedFiles splits by newline and comma', () => {
    const out = normalizeExpectedFiles('a.txt, b.json\nc.bin')
    expect(out).toEqual(['a.txt', 'b.json', 'c.bin'])
  })

  it('formatExpectedFiles joins lines', () => {
    expect(formatExpectedFiles(['a', 'b'])).toBe('a\nb')
    expect(formatExpectedFiles([])).toBe('')
  })

  it('parseMinSizeBytes parses numbers and returns undefined for invalid', () => {
    expect(parseMinSizeBytes('123')).toBe(123)
    expect(parseMinSizeBytes(' 42 ')).toBe(42)
    expect(parseMinSizeBytes('')).toBeUndefined()
    expect(parseMinSizeBytes('   ')).toBeUndefined()
    expect(parseMinSizeBytes('abc')).toBeUndefined()
    expect(parseMinSizeBytes('-1')).toBeUndefined()
    expect(parseMinSizeBytes('1.5')).toBeUndefined()
  })

  it('PROTECTED_MODEL_TYPES contains expected values', () => {
    expect(PROTECTED_MODEL_TYPES.has('semantic_guard')).toBeTruthy()
    expect(PROTECTED_MODEL_TYPES.has('coreference')).toBeTruthy()
  })
})
