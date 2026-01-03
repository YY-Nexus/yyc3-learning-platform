"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import type { User } from "@/app/types"

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // 检查本地存储中的用户信息
    const checkAuth = async () => {
      try {
        const savedUser = localStorage.getItem("user")
        if (savedUser) {
          setUser(JSON.parse(savedUser))
        }
      } catch (error) {
        console.error("检查认证状态失败", error)
      } finally {
        setLoading(false)
      }
    }

    checkAuth()
  }, [])

  const login = async (email: string, password: string) => {
    try {
      // 模拟登录API调用
      await new Promise((resolve) => setTimeout(resolve, 1000))

      const mockUser: User = {
        id: "1",
        name: "张同学",
        email,
        avatar: "/placeholder.svg?height=40&width=40",
        studyPoints: 2450,
        studyDays: 77,
        completedCourses: 12,
        studyHours: 156,
        level: "beginner",
        points: 1500,
        streak: 15,
        joinDate: "2024-01-15",
        certificates: 3,
        rank: 42,
        profile: {
          bio: "",
          location: "",
          website: "",
          github: "",
          linkedin: ""
        },
        learningStats: {
          totalCourses: 15,
          completedCourses: 12,
          totalHours: 156,
          currentStreak: 15,
          longestStreak: 30,
          averageScore: 85.5
        },
        enrolledCourses: [],
        achievements: [],
        preferences: {
          language: "zh-CN",
          timezone: "Asia/Shanghai",
          emailNotifications: true,
          pushNotifications: true,
          weeklyReport: true,
          theme: "system",
          learningReminder: {
            enabled: true,
            time: "09:00",
            days: ["周一", "周二", "周三", "周四", "周五"]
          },
          notifications: true,
          emailUpdates: true
        },
        progress: {}
      }

      setUser(mockUser)
      localStorage.setItem("user", JSON.stringify(mockUser))
    } catch (error) {
      throw new Error("登录失败")
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem("user")
  }

  return <AuthContext.Provider value={{ user, login, logout, loading }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
