import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/globals.css'

// Initialize theme
const theme = localStorage.getItem('dddm-theme') || 'system'
const root = document.documentElement
root.classList.remove('light', 'dark')

if (theme === 'system') {
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  root.classList.add(systemTheme)
} else {
  root.classList.add(theme)
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
) 