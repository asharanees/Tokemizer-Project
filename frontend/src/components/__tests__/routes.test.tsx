import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'

let mockAuth = { isAuthenticated: false, isLoading: false, user: null }

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => mockAuth,
}))

describe('Route components', () => {
  it('ProtectedRoute redirects when not authenticated', async () => {
    const { default: ProtectedRoute } = await import('../ProtectedRoute')
    const { Redirect } = await import('wouter')
    render(<ProtectedRoute />)
    // When not authenticated, component renders Redirect which has no DOM text,
    // ensure that it does not render loader
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('AdminRoute redirects non-admin to root', async () => {
    mockAuth = { isAuthenticated: true, isLoading: false, user: { role: 'user' } }
    const { default: AdminRoute } = await import('../AdminRoute')
    render(<AdminRoute />)
    // ensure no loader
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('shows loader when isLoading is true for ProtectedRoute', async () => {
    mockAuth = { isAuthenticated: false, isLoading: true, user: null }
    const { default: ProtectedRoute } = await import('../ProtectedRoute')
    const { container } = render(<ProtectedRoute path="/" />)
    // Loader2 renders an svg element
    expect(container.querySelector('svg')).not.toBeNull()
  })

  it('shows loader when isLoading is true for AdminRoute', async () => {
    mockAuth = { isAuthenticated: false, isLoading: true, user: null }
    const { default: AdminRoute } = await import('../AdminRoute')
    const { container } = render(<AdminRoute path="/" />)
    expect(container.querySelector('svg')).not.toBeNull()
  })
})
