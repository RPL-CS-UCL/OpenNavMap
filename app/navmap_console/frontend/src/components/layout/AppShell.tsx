import { Outlet } from "react-router-dom";
import { NavRail } from "./NavRail";
import { TopBar } from "./TopBar";

export function AppShell() {
  return (
    <div className="grid h-screen grid-cols-[48px_1fr] grid-rows-[36px_1fr] bg-background text-foreground">
      <NavRail />
      <TopBar />
      <main className="overflow-auto p-4">
        <Outlet />
      </main>
    </div>
  );
}
