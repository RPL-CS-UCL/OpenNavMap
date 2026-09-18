import { Outlet } from "react-router-dom";
import { JobsLive } from "@/features/jobs/JobsLive";
import { JobDock } from "./JobDock";
import { NavRail } from "./NavRail";
import { TopBar } from "./TopBar";

export function AppShell() {
  return (
    <div className="grid h-screen grid-cols-[48px_1fr] grid-rows-[36px_1fr_auto] bg-background text-foreground">
      <NavRail />
      <TopBar />
      <main className="min-h-0 overflow-auto p-4">
        <Outlet />
      </main>
      <JobDock />
      <JobsLive />
    </div>
  );
}
