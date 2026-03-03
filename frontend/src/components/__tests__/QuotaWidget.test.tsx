import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'

vi.mock('@tanstack/react-query', () => ({
  useQuery: vi.fn(),
}))

import { QuotaWidget } from '../layout/QuotaWidget'
import * as rq from '@tanstack/react-query'

describe('QuotaWidget', () => {
  afterEach(() => {
    vi.resetAllMocks()
  })

  it('renders nothing while loading', () => {
    vi.spyOn(rq, 'useQuery').mockReturnValue({ data: undefined, isLoading: true })
    const { container } = render(<QuotaWidget />)
    expect(container.firstChild).toBeNull()
  })

  it('renders usage and badge when data present (non-critical)', () => {
    const data = {
      calls_used: 10,
      quota_limit: 100,
      remaining: 90,
      subscription_tier: 'pro',
    }
    vi.spyOn(rq, 'useQuery').mockReturnValue({ data, isLoading: false })
    render(<QuotaWidget />)
    expect(screen.getByText('Usage')).toBeInTheDocument()
    expect(screen.getByText('PRO')).toBeInTheDocument()
    expect(screen.queryByText('Quota nearly exhausted!')).toBeNull()
  })

  it('shows critical message when usage >80%', () => {
    const data = {
      calls_used: 900,
      quota_limit: 1000,
      remaining: 100,
      subscription_tier: 'team',
    }
    vi.spyOn(rq, 'useQuery').mockReturnValue({ data, isLoading: false })
    render(<QuotaWidget />)
    expect(screen.getByText('Quota nearly exhausted!')).toBeInTheDocument()
    expect(screen.getByText('TEAM')).toBeInTheDocument()
  })
})
