"use client"

import React, { useState, useEffect } from "react"
import * as XLSX from "xlsx"
import { FileText, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ExcelPreviewProps {
  fileUrl: string
}

export default function ExcelPreview({ fileUrl }: ExcelPreviewProps) {
  const [sheets, setSheets] = useState<{ [key: string]: any[][] }>({})
  const [sheetNames, setSheetNames] = useState<string[]>([])
  const [activeSheet, setActiveSheet] = useState<string>("")
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    const loadExcel = async () => {
      try {
        const response = await fetch(fileUrl)
        if (!response.ok) throw new Error("Failed to fetch Excel file")
        
        const arrayBuffer = await response.arrayBuffer()
        const workbook = XLSX.read(arrayBuffer, { type: "array" })
        
        const sheetsData: { [key: string]: any[][] } = {}
        workbook.SheetNames.forEach((sheetName) => {
          const worksheet = workbook.Sheets[sheetName]
          // Convert to 2D array with headers
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 })
          sheetsData[sheetName] = jsonData as any[][]
        })
        
        setSheets(sheetsData)
        setSheetNames(workbook.SheetNames)
        setActiveSheet(workbook.SheetNames[0])
      } catch (err) {
        console.error("Error loading Excel:", err)
        setError("Failed to load Excel file")
      } finally {
        setIsLoading(false)
      }
    }

    loadExcel()
  }, [fileUrl])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-text-secondary">Loading Excel file...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-text-tertiary">
          <FileText className="w-16 h-16 mx-auto mb-4 text-red-400" />
          <p className="text-lg font-semibold text-red-600">Error Loading Excel</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    )
  }

  const currentSheetData = sheets[activeSheet] || []
  const currentIndex = sheetNames.indexOf(activeSheet)

  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      {/* Sheet tabs */}
      {sheetNames.length > 1 && (
        <div className="bg-bg-card border-b border-border-color px-4 py-2 flex items-center gap-2 flex-shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setActiveSheet(sheetNames[Math.max(0, currentIndex - 1)])}
            disabled={currentIndex === 0}
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          
          <div className="flex gap-2 flex-1 overflow-x-auto">
            {sheetNames.map((name) => (
              <button
                key={name}
                onClick={() => setActiveSheet(name)}
                className={`px-3 py-1 text-sm rounded transition-colors whitespace-nowrap ${
                  activeSheet === name
                    ? "bg-primary text-primary-foreground"
                    : "bg-bg-subtle text-text-secondary hover:bg-bg-hover"
                }`}
              >
                {name}
              </button>
            ))}
          </div>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setActiveSheet(sheetNames[Math.min(sheetNames.length - 1, currentIndex + 1)])}
            disabled={currentIndex === sheetNames.length - 1}
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      )}
      
      {/* Spreadsheet table */}
      <div className="flex-1 overflow-auto">
        <div className="min-w-max">
          <table className="w-full border-collapse">
            <tbody>
              {currentSheetData.map((row, rowIndex) => (
                <tr key={rowIndex} className={rowIndex === 0 ? "bg-bg-subtle font-semibold" : ""}>
                  {(row as any[]).map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="border border-border-color px-3 py-2 text-sm min-w-[100px] max-w-[400px] break-words"
                    >
                      {cell !== null && cell !== undefined ? String(cell) : ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
