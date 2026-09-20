import { Component, type ErrorInfo, type ReactNode } from "react";
import { t } from "@/i18n";
import { Button } from "@/components/ui/button";

interface Props { children: ReactNode; onReset?: () => void }
interface State { error: Error | null }

export class SceneErrorBoundary extends Component<Props, State> {
  state: State = { error: null };
  static getDerivedStateFromError(error: Error): State { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error("scene crashed", error, info.componentStack); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
        <span>{t("scene.crashed")}</span>
        <code className="max-w-md truncate text-xs">{this.state.error.message}</code>
        <Button size="sm" variant="outline" onClick={() => { this.setState({ error: null }); this.props.onReset?.(); }}>{t("scene.reload")}</Button>
      </div>
    );
  }
}
