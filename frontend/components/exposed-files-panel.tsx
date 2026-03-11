"use client";

import React from "react";
import { Download, File, FileText, Image, FileCode, Archive, ExternalLink, FileSpreadsheet } from "lucide-react";
import { ExposedFile } from "@/lib/types";

interface ExposedFilesPanelProps {
    files: ExposedFile[];
}

const getFileIcon = (type: ExposedFile["type"], className = "w-4 h-4") => {
    switch (type) {
        case "image": return <Image className={className} />;
        case "document": return <FileText className={className} />;
        case "spreadsheet": return <FileSpreadsheet className={className} />;
        case "code": return <FileCode className={className} />;
        case "archive": return <Archive className={className} />;
        default: return <File className={className} />;
    }
};

const formatBytes = (bytes?: number) => {
    if (bytes === undefined || bytes === null) return "Unknown size";
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
};

export const ExposedFilesPanel: React.FC<ExposedFilesPanelProps> = ({ files }) => {
    if (!files || files.length === 0) return null;

    return (
        <div className="mt-4 flex flex-col gap-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                Attached Files
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {files.map((file) => {
                    // Determine best URL to use (files endpoint handles serving properly)
                    const downloadUrl = file.path.startsWith("/") ? file.path : `/files/${file.path}`;
                    const isPreviewable = ["image", "document", "code"].includes(file.type);

                    return (
                        <div
                            key={file.id}
                            className="flex items-center justify-between p-3 border rounded-xl bg-white shadow-sm hover:shadow-md transition-shadow dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800"
                        >
                            <div className="flex items-center gap-3 overflow-hidden">
                                <div className="p-2 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-500 rounded-lg flex-shrink-0">
                                    {getFileIcon(file.type, "w-5 h-5")}
                                </div>
                                <div className="flex flex-col min-w-0">
                                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate" title={file.name}>
                                        {file.name}
                                    </span>
                                    <span className="text-xs text-gray-500 truncate" title={file.description || formatBytes(file.size_bytes)}>
                                        {file.description || formatBytes(file.size_bytes)}
                                    </span>
                                </div>
                            </div>

                            <div className="flex items-center gap-1 ml-2 flex-shrink-0">
                                {isPreviewable && (
                                    <a
                                        href={downloadUrl}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="p-1.5 text-gray-400 hover:text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/40 rounded-md transition-colors"
                                        title="Open / Preview"
                                    >
                                        <ExternalLink className="w-4 h-4" />
                                    </a>
                                )}
                                <a
                                    href={downloadUrl}
                                    download={file.name}
                                    className="p-1.5 text-gray-400 hover:text-green-600 hover:bg-green-50 dark:hover:bg-green-900/40 rounded-md transition-colors"
                                    title="Download File"
                                >
                                    <Download className="w-4 h-4 cursor-pointer" />
                                </a>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
