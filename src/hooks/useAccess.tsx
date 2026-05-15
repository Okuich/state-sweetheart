/**
 * Reactive access snapshot (roles + feature flags) for the merged platform.
 * Backed by `getMyAccess` server fn; refreshes on auth state change.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useServerFn } from "@tanstack/react-start";
import { supabase } from "@/integrations/supabase/client";
import { getMyAccess, type AccessSnapshot, type AppRole } from "@/lib/access.functions";
import { useAuth } from "@/hooks/useAuth";

interface AccessCtx {
  snapshot: AccessSnapshot | null;
  loading: boolean;
  refresh: () => Promise<void>;
  hasRole: (role: AppRole) => boolean;
  hasAnyRole: (roles: AppRole[]) => boolean;
  hasFlag: (key: string) => boolean;
  isAdmin: boolean;
}

const Ctx = createContext<AccessCtx>({
  snapshot: null,
  loading: true,
  refresh: async () => {},
  hasRole: () => false,
  hasAnyRole: () => false,
  hasFlag: () => false,
  isAdmin: false,
});

export function AccessProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const fetchAccess = useServerFn(getMyAccess);
  const [snapshot, setSnapshot] = useState<AccessSnapshot | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!user) {
      setSnapshot(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const data = await fetchAccess();
      setSnapshot(data);
    } catch {
      setSnapshot(null);
    } finally {
      setLoading(false);
    }
  }, [user, fetchAccess]);

  useEffect(() => {
    refresh();
    const { data: { subscription } } = supabase.auth.onAuthStateChange(() => {
      refresh();
    });
    return () => subscription.unsubscribe();
  }, [refresh]);

  const value = useMemo<AccessCtx>(() => {
    const roles = snapshot?.roles ?? [];
    const flags = snapshot?.flags ?? {};
    return {
      snapshot,
      loading,
      refresh,
      hasRole: (r) => roles.includes(r),
      hasAnyRole: (rs) => rs.some((r) => roles.includes(r)),
      hasFlag: (k) => !!flags[k],
      isAdmin: roles.includes("admin"),
    };
  }, [snapshot, loading, refresh]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAccess() {
  return useContext(Ctx);
}

export function usePhysicsAccess() {
  const a = useAccess();
  return { hasAccess: a.hasFlag("physics_engine"), loading: a.loading };
}
