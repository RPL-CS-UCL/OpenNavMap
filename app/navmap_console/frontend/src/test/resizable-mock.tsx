import { useRef, type CSSProperties, type ReactNode } from "react";

// jsdom cannot measure panel sizes, so react-resizable-panels computes "NaN%" flex-basis
// values that make jsdom's CSS parser throw. Tests render a plain flex layout instead;
// drag-resize behaviour is not testable in jsdom anyway.
// react-resizable-panels v4 导出的是 Group / Panel / Separator（ui/resizable.tsx 按这三个名字导入）。
export function Group({ children, className, orientation = "horizontal" }: {
  children: ReactNode;
  className?: string;
  orientation?: "horizontal" | "vertical";
}) {
  return (
    <div
      className={className}
      style={{
        display: "flex",
        flexDirection: orientation === "vertical" ? "column" : "row",
        minWidth: 0,
        minHeight: 0,
      }}
    >
      {children}
    </div>
  );
}

export function Panel({ children, className, style }: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  defaultSize?: number | string;
  minSize?: number | string;
  maxSize?: number | string;
  collapsible?: boolean;
  collapsedSize?: number | string;
  panelRef?: unknown;
  order?: number;
  id?: string;
}) {
  return (
    <div className={className} style={style}>
      {children}
    </div>
  );
}

export function Separator({ className }: { className?: string }) {
  return <div className={className} />;
}

/** No layout in jsdom: the imperative handle is a stand-in that only tracks the collapsed flag. */
export function usePanelRef() {
  const collapsed = useRef(false);
  return useRef({
    collapse: () => { collapsed.current = true; },
    expand: () => { collapsed.current = false; },
    isCollapsed: () => collapsed.current,
  });
}
