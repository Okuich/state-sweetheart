/**
 * Customer-facing Fabrication OS dashboard.
 *
 * Physics OS and Geometry OS are intentionally NOT surfaced here —
 * they are hidden internal layers. The Fab dashboard is the only
 * customer-visible workspace.
 */
import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect } from "react";
import { useAccess } from "@/hooks/useAccess";
import { Button } from "@/components/ui/button";
import { FabFeedbackPanel } from "@/components/FabFeedbackPanel";
import { ScanImportPanel } from "@/components/ScanImportPanel";
import { MaterialEditorPanel } from "@/components/MaterialEditorPanel";

import { FabPredictionsPanel } from "@/components/FabPredictionsPanel";
import { startPhysicsFabBridge } from "@/lib/physicsFabBridge";

export const Route = createFileRoute("/_authenticated/")({
  component: FabricationDashboard,
  head: () => ({
    meta: [
      { title: "Fabrication OS" },
      {
        name: "description",
        content:
          "Fabrication OS — scan ingest, calibration feedback, materials and templates for production parts.",
      },
    ],
  }),
});

function FabricationDashboard() {
  const { isAdmin } = useAccess();

  // Boot the hidden Physics OS → Fab OS bridge once.
  useEffect(() => {
    const stop = startPhysicsFabBridge();
    return () => stop();
  }, []);

  return (
    <main className="relative min-h-screen px-6 py-10 lg:px-10">
      <header className="mb-10 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="h-2.5 w-2.5 rounded-full bg-primary animate-pulse" />
            <span className="text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
              Fabrication OS
            </span>
          </div>
          <h1 className="mt-3 font-display text-4xl font-bold leading-[0.95] md:text-5xl">
            Production-grade
            <br />
            <span className="text-primary">parts and parts-of-parts.</span>
          </h1>
          <p className="mt-3 max-w-xl text-sm text-muted-foreground">
            Import inspection reports, calibrate against predictions, and
            manage the materials and templates your fab line consumes.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline" size="sm">
            <Link to="/jobs">Jobs</Link>
          </Button>
          <Button asChild variant="outline" size="sm">
            <Link to="/api-keys">API keys</Link>
          </Button>
          {isAdmin && (
            <Button asChild variant="ghost" size="sm" className="text-[10px] uppercase tracking-[0.22em]">
              <Link to="/internal">Internal console</Link>
            </Button>
          )}
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <ScanImportPanel />
        <FabFeedbackPanel />
      </div>

      <div className="mt-6">
        <FabPredictionsPanel />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <MaterialEditorPanel />
        <TemplatesPanel />
      </div>
    </main>
  );
}
