import { test, expect, type ConsoleMessage } from "@playwright/test";

/**
 * Smoke test: load "/" and verify the headline UI renders without any
 * runtime errors firing in the browser.
 *
 * "Runtime errors" = console.error entries OR uncaught page exceptions.
 * We allow a small allowlist for noisy-but-harmless dev warnings (e.g.
 * WebGPU adapter not available in CI Chromium without --enable-unsafe-webgpu).
 */

const ERROR_ALLOWLIST: RegExp[] = [
  /WebGPU/i,
  /navigator\.gpu/i,
  /requestAdapter/i,
  /Failed to load resource.*favicon/i,
  // React DevTools nag in dev
  /Download the React DevTools/i,
];

function isAllowed(msg: string): boolean {
  return ERROR_ALLOWLIST.some((re) => re.test(msg));
}

test.describe("app smoke", () => {
  test("home route renders key UI without runtime errors", async ({ page }) => {
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];

    page.on("console", (m: ConsoleMessage) => {
      if (m.type() !== "error") return;
      const text = m.text();
      if (!isAllowed(text)) consoleErrors.push(text);
    });
    page.on("pageerror", (err) => {
      const text = err.message ?? String(err);
      if (!isAllowed(text)) pageErrors.push(text);
    });

    const response = await page.goto("/", { waitUntil: "domcontentloaded" });
    expect(response, "navigation response").not.toBeNull();
    expect(response!.status(), "HTTP status").toBeLessThan(400);

    // Header brand mark.
    await expect(page.getByText("PhysicsState", { exact: false })).toBeVisible();

    // Device / dtype telemetry chip in header.
    await expect(page.getByText(/device\s*·/i)).toBeVisible();
    await expect(page.getByText(/dtype\s*·/i)).toBeVisible();

    // The simulation canvas mounts via PhysicsCanvas.
    await expect(page.locator("canvas").first()).toBeVisible();

    // A representative panel from the dashboard. Use a stable label.
    await expect(
      page.getByText(/GPU precision policy/i).first(),
    ).toBeVisible();

    // Give the app a beat to flush any deferred effects / async errors.
    await page.waitForTimeout(1000);

    expect(
      pageErrors,
      `uncaught page errors:\n${pageErrors.join("\n")}`,
    ).toEqual([]);
    expect(
      consoleErrors,
      `console.error messages:\n${consoleErrors.join("\n")}`,
    ).toEqual([]);
  });

  test("why-this-matters precision popover opens", async ({ page }) => {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    const trigger = page.getByRole("button", { name: /why this matters/i });
    await expect(trigger).toBeVisible();
    await trigger.click();
    await expect(page.getByText(/f32 vs f64/i)).toBeVisible();
    await expect(page.getByText(/Energy drift/i)).toBeVisible();
    await expect(page.getByText(/Determinism/i)).toBeVisible();
  });
});
