import React from 'react'
import { render, screen } from '@testing-library/react'
import { Sidebar } from '../Sidebar'

// Create a minimal AuthContext mock
import { vi } from 'vitest'

let mockRole = 'user'

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({ user: { role: mockRole } }),
}))

vi.mock('@tanstack/react-query', () => ({
  useQuery: vi.fn(() => ({ data: null, isLoading: false })),
}))

vi.mock('./QuotaWidget', () => ({
  QuotaWidget: () => <div>Quota Widget</div>,
}))

describe('Sidebar', () => {
  it('renders navigation items', () => {
    render(<Sidebar />)
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Optimizer')).toBeInTheDocument()
  })

  it('shows admin items only for admin role', async () => {
    mockRole = 'admin'
    render(<Sidebar />)
    expect(screen.getByText('Manage Users')).toBeInTheDocument()
    expect(screen.getByText('Model Management')).toBeInTheDocument()
    mockRole = 'user'
  })
})
