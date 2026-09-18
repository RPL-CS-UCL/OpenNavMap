import type { ReactNode } from "react";
import { Navigate, createBrowserRouter } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { EmptyState } from "@/components/common/EmptyState";
import { RouteError } from "@/components/common/RouteError";
import { RegionCrumb, SessionCrumb } from "@/components/layout/crumbs";
import { JobDetailPage } from "@/features/jobs/JobDetailPage";
import { JobsPage } from "@/features/jobs/JobsPage";
import { RegionDetailPage } from "@/features/regions/RegionDetailPage";
import { RegionsPage } from "@/features/regions/RegionsPage";
import { NewSessionPage } from "@/features/sessions/NewSessionPage";
import { SessionDetailPage } from "@/features/sessions/SessionDetailPage";
import { t } from "@/i18n";
import { Inbox, Settings } from "lucide-react";

export type RouteParams = Record<string, string | undefined>;
export interface CrumbHandle {
  crumb: (params: RouteParams) => ReactNode;
}

const crumb = (fn: CrumbHandle["crumb"]): CrumbHandle => ({ crumb: fn });

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Navigate to="/regions" replace /> },
      {
        path: "regions",
        handle: crumb(() => t("nav.regions")),
        children: [
          { index: true, element: <RegionsPage />, errorElement: <RouteError /> },
          {
            path: ":rid",
            handle: crumb((p) => <RegionCrumb rid={p.rid ?? ""} />),
            children: [
              { index: true, element: <RegionDetailPage />, errorElement: <RouteError /> },
              {
                path: "sessions/new",
                element: <NewSessionPage />,
                errorElement: <RouteError />,
                handle: crumb(() => t("session.new.title")),
              },
              {
                path: "sessions/:sid",
                element: <SessionDetailPage />,
                errorElement: <RouteError />,
                handle: crumb((p) => <SessionCrumb rid={p.rid ?? ""} sid={p.sid ?? ""} />),
              },
            ],
          },
        ],
      },
      {
        path: "jobs",
        handle: crumb(() => t("nav.jobs")),
        children: [
          { index: true, element: <JobsPage />, errorElement: <RouteError /> },
          { path: ":jid", element: <JobDetailPage />, errorElement: <RouteError />, handle: crumb((p) => p.jid ?? "") },
        ],
      },
      {
        path: "settings",
        handle: crumb(() => t("nav.settings")),
        element: <EmptyState icon={Settings} title={t("nav.settings")} body={t("settings.placeholder")} />,
      },
      { path: "*", element: <EmptyState icon={Inbox} title={t("common.notFound")} /> },
    ],
  },
]);
