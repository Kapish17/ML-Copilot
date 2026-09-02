import { DataTable, Td, Th } from "@/components/common/DataTable";
import type { ClassificationDetails } from "@/lib/api/types";
import { formatCount } from "@/lib/format";

/**
 * Predicted against actual, as a real table.
 *
 * A confusion matrix is a table of counts, so it is marked up as one, with a
 * row header per actual class and a column header per predicted class. That
 * makes a screen reader announce "actual yes, predicted no, 1" instead of a
 * bare number in a grid — which is the whole content of the cell.
 *
 * Correct cells are shaded, but the diagonal is also stated in the caption,
 * so the shading is a convenience rather than the information.
 */
export function ConfusionMatrix({
  details,
}: {
  details: ClassificationDetails;
}) {
  const { class_labels: labels, confusion_matrix: matrix } = details;
  if (!labels?.length || !matrix?.length) return null;

  return (
    <div>
      <DataTable
        visibleCaption
        caption="Rows are the actual class, columns the predicted class. Cells on the diagonal are correct predictions."
        head={
          <tr>
            <Th>Actual \ Predicted</Th>
            {labels.map((label) => (
              <Th key={label} numeric>
                {label}
              </Th>
            ))}
          </tr>
        }
      >
        {matrix.map((row, rowIndex) => (
          <tr key={labels[rowIndex] ?? rowIndex}>
            <Th
              scope="row"
              className="normal-case tracking-normal text-ink-900"
            >
              {labels[rowIndex] ?? rowIndex}
            </Th>
            {row.map((count, columnIndex) => (
              <Td
                key={`${rowIndex}-${columnIndex}`}
                numeric
                className={
                  rowIndex === columnIndex
                    ? "bg-emerald-50 font-semibold text-emerald-900"
                    : count > 0
                      ? "text-rose-800"
                      : "text-ink-400"
                }
              >
                {formatCount(count)}
              </Td>
            ))}
          </tr>
        ))}
      </DataTable>

      {details.positive_label && (
        <p className="mt-2 text-xs text-ink-500">
          Positive label:{" "}
          <span className="font-mono">{details.positive_label}</span> · averaging:{" "}
          {details.averaging}
        </p>
      )}
    </div>
  );
}
