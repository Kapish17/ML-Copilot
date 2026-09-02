"use client";

import { useId, useRef, useState } from "react";
import { Button } from "@/components/common/Button";
import { Badge } from "@/components/common/Badge";
import { SUPPORTED_EXTENSIONS, UPLOAD_ACCEPT } from "@/lib/api/datasets";
import { formatBytes } from "@/lib/format";

/**
 * Choosing the file everything else on the dashboard works from.
 *
 * The extension check here is a courtesy, not a gate. The backend is the
 * authority on what it can read — it looks at the bytes, not the name — so
 * this only saves a round trip on an obviously wrong file, and says so in the
 * same words the backend would use. Anything it lets through is still the
 * backend's decision.
 *
 * The file itself never leaves this component except as a `File` handed to
 * the API client, which posts it to the configured backend and nowhere else.
 * It is not read into state, not serialised, and not written to any browser
 * storage.
 */
export interface DatasetUploadProps {
  file: File | null;
  onSelect: (file: File | null) => void;
  disabled?: boolean;
}

function hasSupportedExtension(name: string): boolean {
  const lower = name.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

export function DatasetUpload({
  file,
  onSelect,
  disabled = false,
}: DatasetUploadProps) {
  const inputId = useId();
  const hintId = `${inputId}-hint`;
  const errorId = `${inputId}-error`;
  const inputRef = useRef<HTMLInputElement>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  function accept(candidate: File | null) {
    if (!candidate) {
      setLocalError(null);
      onSelect(null);
      return;
    }
    if (!hasSupportedExtension(candidate.name)) {
      setLocalError(
        "That file type is not supported. Upload a CSV, an Excel workbook (.xlsx) or a JSON file.",
      );
      onSelect(null);
      return;
    }
    setLocalError(null);
    onSelect(candidate);
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        className="block text-sm font-medium text-ink-800"
      >
        Dataset file
      </label>

      <div
        onDragOver={(event) => {
          if (disabled) return;
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          if (disabled) return;
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files?.[0] ?? null);
        }}
        className={`mt-2 rounded-lg border-2 border-dashed px-4 py-5 transition-colors ${
          dragging ? "border-accent-400 bg-accent-50" : "border-ink-200 bg-ink-50"
        }`}
      >
        <input
          ref={inputRef}
          id={inputId}
          name="file"
          type="file"
          accept={UPLOAD_ACCEPT}
          disabled={disabled}
          aria-describedby={localError ? `${hintId} ${errorId}` : hintId}
          onChange={(event) => accept(event.target.files?.[0] ?? null)}
          className="block w-full text-sm text-ink-700 file:mr-3 file:rounded-md file:border-0 file:bg-ink-900 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-ink-800 disabled:opacity-60"
        />

        <p id={hintId} className="mt-3 text-xs text-ink-600">
          Supported: <span className="font-medium">CSV · Excel (.xlsx) · JSON</span>.
          Excel reads the first worksheet; JSON must be an array of objects, one
          per row. The file is sent to the ML Copilot backend for this request
          only and is never stored in your browser.
        </p>

        {file && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Badge tone="accent" glyph="▣">
              {file.name}
            </Badge>
            <span className="text-xs text-ink-500">{formatBytes(file.size)}</span>
            <Button
              variant="ghost"
              className="px-2 py-1 text-xs"
              disabled={disabled}
              onClick={() => {
                if (inputRef.current) inputRef.current.value = "";
                accept(null);
              }}
            >
              Remove
            </Button>
          </div>
        )}
      </div>

      {localError && (
        <p id={errorId} role="alert" className="mt-2 text-sm text-rose-700">
          {localError}
        </p>
      )}
    </div>
  );
}
