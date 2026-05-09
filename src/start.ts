/**
 * Global TanStack Start configuration.
 *
 * `attachAuth` runs on the client before every server-fn RPC and attaches
 * the current Supabase access token as a Bearer header so server functions
 * gated by `requireSupabaseAuth` can identify the user.
 */
import { createStart, createMiddleware } from "@tanstack/react-start";
import { supabase } from "@/integrations/supabase/client";

const attachAuth = createMiddleware({ type: "function" }).client(async ({ next }) => {
  let token: string | null = null;
  if (typeof window !== "undefined") {
    const { data } = await supabase.auth.getSession();
    token = data.session?.access_token ?? null;
  }
  return next(token ? { headers: { Authorization: `Bearer ${token}` } } : {});
});

export const startInstance = createStart(() => ({
  functionMiddleware: [attachAuth],
}));
