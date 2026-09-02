/** Dataset endpoints: profiling an upload. */

import { postForm, type RequestOptions } from "./client";
import type { DatasetProfile } from "./types";

/** The formats the upload control offers, mirroring the backend allowlist. */
export const SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".json"] as const;

/** The `accept` attribute for a file input, covering names and media types. */
export const UPLOAD_ACCEPT =
  ".csv,.xlsx,.json,text/csv,application/json," +
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

/**
 * Profile an uploaded dataset.
 *
 * @param file - The dataset. CSV, Excel (.xlsx) or JSON; the backend decides.
 * @param targetColumn - Optional column to analyse as the target.
 * @returns The complete profile, including which format the file was read as.
 */
export function profileDataset(
  file: File,
  targetColumn?: string,
  options: RequestOptions = {},
): Promise<DatasetProfile> {
  const form = new FormData();
  form.append("file", file);
  if (targetColumn) form.append("target_column", targetColumn);
  return postForm<DatasetProfile>("/api/v1/datasets/profile", form, options);
}
