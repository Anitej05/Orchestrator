"use client"

import React, { useState, useEffect } from "react"
import Papa from "papaparse"
import { FileText } from "lucide-react"

interface CsvPreviewProps {
  fileUrl: string
}

export default function CsvPreview({ fileUrl }: CsvPreviewProps) {
  const [data, setData] = useState<string[][]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    const loadCsv = async () => {
      try {
        const response = await fetch(fileUrl)
        if (!response.ok) throw new Error("Failed to fetch CSV file")
        
        const text = await response.text()
        
        Papa.parse(text, {
          complete: (results) => {
            setData(results.data as string[][])
            setIsLoading(false)
          },
          error: (err: Error) => {
            console.error("Error parsing CSV:", err)
            setError("Failed to parse CSV file")
            setIsLoading(false)
          }
        })
      } catch (err) {
        console.error("Error loading CSV:", err)
        setError("Failed to load CSV file")
        setIsLoading(false)
      }
    }

    loadCsv()
  }, [fileUrl])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-text-secondary">Loading CSV...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-text-tertiary">
          <FileText className="w-16 h-16 mx-auto mb-4 text-red-400" />
          <p className="text-lg font-semibold text-red-600">Error Loading CSV</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full overflow-auto">
      <div className="min-w-max">
        <table className="w-full border-collapse">
          <tbody>
            {data.map((row, rowIndex) => (
              <tr key={rowIndex} className={rowIndex === 0 ? "bg-bg-subtle font-semibold" : ""}>
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="border border-border-color px-3 py-2 text-sm min-w-[100px] max-w-[400px] break-words"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
