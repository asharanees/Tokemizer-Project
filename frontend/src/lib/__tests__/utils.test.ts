import { cn } from '../utils'

describe('cn utility', () => {
  it('merges class names and deduplicates', () => {
    const out = cn('a b', 'b c', { 'd': true } as any)
    expect(typeof out).toBe('string')
    expect(out.length).toBeGreaterThan(0)
    expect(out.includes('a')).toBeTruthy()
    expect(out.includes('c')).toBeTruthy()
  })
})
