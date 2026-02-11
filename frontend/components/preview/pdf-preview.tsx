"use client"

import React from "react"
import { Viewer, Worker } from "@react-pdf-viewer/core"
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout"
import "@react-pdf-viewer/core/lib/styles/index.css"
import "@react-pdf-viewer/default-layout/lib/styles/index.css"

interface PdfPreviewProps {
  fileUrl: string
}

export default function PdfPreview({ fileUrl }: PdfPreviewProps) {
  const defaultLayoutPluginInstance = defaultLayoutPlugin()

  return (
    <div className="h-full w-full overflow-hidden">
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
        <div className="h-full">
          <Viewer
            fileUrl={fileUrl}
            plugins={[defaultLayoutPluginInstance]}
          />
        </div>
      </Worker>
    </div>
  )
}
