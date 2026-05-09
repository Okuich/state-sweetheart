import { createServerOnlyFn } from "@tanstack/react-start";
import { supabaseAdmin } from "@/integrations/supabase/client.server";

/**
 * Server-only accessor for the Supabase admin client.
 * Body is stripped from client bundles by the start compiler, so the
 * `client.server` import does not leak past `import-protection`.
 */
export const getAdmin = createServerOnlyFn(() => supabaseAdmin);
