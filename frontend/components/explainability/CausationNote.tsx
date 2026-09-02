/**
 * The sentence that has to appear next to every attribution on this site.
 *
 * SHAP answers "how did this model respond to this input", which reads very
 * easily as "what caused the outcome" and is not the same claim. The note is
 * a component rather than a string so it cannot be forgotten in one place and
 * present in another.
 */
export function CausationNote({ className = "" }: { className?: string }) {
  return (
    <p className={`text-xs text-ink-500 ${className}`}>
      Feature importance describes model behaviour and association, not
      causation.
    </p>
  );
}
