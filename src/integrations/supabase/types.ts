export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      api_clients: {
        Row: {
          created_at: string
          created_by: string | null
          id: string
          key_hash: string
          key_prefix: string
          last_used_at: string | null
          name: string
          revoked_at: string | null
          scopes: string[]
        }
        Insert: {
          created_at?: string
          created_by?: string | null
          id?: string
          key_hash: string
          key_prefix: string
          last_used_at?: string | null
          name: string
          revoked_at?: string | null
          scopes?: string[]
        }
        Update: {
          created_at?: string
          created_by?: string | null
          id?: string
          key_hash?: string
          key_prefix?: string
          last_used_at?: string | null
          name?: string
          revoked_at?: string | null
          scopes?: string[]
        }
        Relationships: []
      }
      api_request_log: {
        Row: {
          client_id: string | null
          created_at: string
          id: number
          latency_ms: number | null
          method: string
          route: string
          status: number
        }
        Insert: {
          client_id?: string | null
          created_at?: string
          id?: number
          latency_ms?: number | null
          method: string
          route: string
          status: number
        }
        Update: {
          client_id?: string | null
          created_at?: string
          id?: number
          latency_ms?: number | null
          method?: string
          route?: string
          status?: number
        }
        Relationships: [
          {
            foreignKeyName: "api_request_log_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "api_clients"
            referencedColumns: ["id"]
          },
        ]
      }
      feature_flags: {
        Row: {
          allowed_roles: string[]
          created_at: string
          description: string | null
          enabled: boolean
          key: string
          metadata: Json
          rollout_percentage: number
          updated_at: string
        }
        Insert: {
          allowed_roles?: string[]
          created_at?: string
          description?: string | null
          enabled?: boolean
          key: string
          metadata?: Json
          rollout_percentage?: number
          updated_at?: string
        }
        Update: {
          allowed_roles?: string[]
          created_at?: string
          description?: string | null
          enabled?: boolean
          key?: string
          metadata?: Json
          rollout_percentage?: number
          updated_at?: string
        }
        Relationships: []
      }
      materials: {
        Row: {
          cost_usd_per_kg: number | null
          created_at: string
          density_kg_m3: number | null
          family: string
          id: string
          metadata: Json
          name: string
          slug: string
          thermal_conductivity_w_mk: number | null
          ultimate_strength_mpa: number | null
          updated_at: string
          yield_strength_mpa: number | null
          youngs_modulus_gpa: number | null
        }
        Insert: {
          cost_usd_per_kg?: number | null
          created_at?: string
          density_kg_m3?: number | null
          family: string
          id?: string
          metadata?: Json
          name: string
          slug: string
          thermal_conductivity_w_mk?: number | null
          ultimate_strength_mpa?: number | null
          updated_at?: string
          yield_strength_mpa?: number | null
          youngs_modulus_gpa?: number | null
        }
        Update: {
          cost_usd_per_kg?: number | null
          created_at?: string
          density_kg_m3?: number | null
          family?: string
          id?: string
          metadata?: Json
          name?: string
          slug?: string
          thermal_conductivity_w_mk?: number | null
          ultimate_strength_mpa?: number | null
          updated_at?: string
          yield_strength_mpa?: number | null
          youngs_modulus_gpa?: number | null
        }
        Relationships: []
      }
      patent_records: {
        Row: {
          claims: Json
          created_at: string
          id: string
          jurisdiction: string | null
          metadata: Json
          status: string
          title: string
          updated_at: string
          user_id: string
        }
        Insert: {
          claims?: Json
          created_at?: string
          id?: string
          jurisdiction?: string | null
          metadata?: Json
          status?: string
          title: string
          updated_at?: string
          user_id: string
        }
        Update: {
          claims?: Json
          created_at?: string
          id?: string
          jurisdiction?: string | null
          metadata?: Json
          status?: string
          title?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      physics_jobs: {
        Row: {
          created_at: string
          duration_ms: number | null
          error: string | null
          id: string
          input: Json
          kind: string
          result: Json | null
          status: string
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          duration_ms?: number | null
          error?: string | null
          id?: string
          input: Json
          kind: string
          result?: Json | null
          status?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          duration_ms?: number | null
          error?: string | null
          id?: string
          input?: Json
          kind?: string
          result?: Json | null
          status?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      pilot_engagements: {
        Row: {
          company: string
          contact_email: string
          created_at: string
          id: string
          metadata: Json
          notes: string | null
          stage: string
          updated_at: string
          user_id: string
        }
        Insert: {
          company: string
          contact_email: string
          created_at?: string
          id?: string
          metadata?: Json
          notes?: string | null
          stage?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          company?: string
          contact_email?: string
          created_at?: string
          id?: string
          metadata?: Json
          notes?: string | null
          stage?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          display_name: string | null
          id: string
          updated_at: string
          user_id: string
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          display_name?: string | null
          id?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          display_name?: string | null
          id?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      role_requests: {
        Row: {
          created_at: string
          id: string
          reason: string | null
          requested_flag: string | null
          requested_role: Database["public"]["Enums"]["app_role"]
          reviewed_at: string | null
          reviewed_by: string | null
          status: string
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          reason?: string | null
          requested_flag?: string | null
          requested_role?: Database["public"]["Enums"]["app_role"]
          reviewed_at?: string | null
          reviewed_by?: string | null
          status?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          reason?: string | null
          requested_flag?: string | null
          requested_role?: Database["public"]["Enums"]["app_role"]
          reviewed_at?: string | null
          reviewed_by?: string | null
          status?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      step_job_events: {
        Row: {
          created_at: string
          data: Json | null
          id: number
          job_id: string
          message: string | null
          progress: number
          stage: string
        }
        Insert: {
          created_at?: string
          data?: Json | null
          id?: number
          job_id: string
          message?: string | null
          progress?: number
          stage: string
        }
        Update: {
          created_at?: string
          data?: Json | null
          id?: number
          job_id?: string
          message?: string | null
          progress?: number
          stage?: string
        }
        Relationships: [
          {
            foreignKeyName: "step_job_events_job_id_fkey"
            columns: ["job_id"]
            isOneToOne: false
            referencedRelation: "step_jobs"
            referencedColumns: ["id"]
          },
        ]
      }
      step_jobs: {
        Row: {
          client_id: string | null
          completed_at: string | null
          created_at: string
          error: string | null
          filename: string
          geometry: Json | null
          id: string
          mesh: Json | null
          progress: Json | null
          reasoning: Json | null
          status: string
          storage_path: string
        }
        Insert: {
          client_id?: string | null
          completed_at?: string | null
          created_at?: string
          error?: string | null
          filename: string
          geometry?: Json | null
          id?: string
          mesh?: Json | null
          progress?: Json | null
          reasoning?: Json | null
          status?: string
          storage_path: string
        }
        Update: {
          client_id?: string | null
          completed_at?: string | null
          created_at?: string
          error?: string | null
          filename?: string
          geometry?: Json | null
          id?: string
          mesh?: Json | null
          progress?: Json | null
          reasoning?: Json | null
          status?: string
          storage_path?: string
        }
        Relationships: [
          {
            foreignKeyName: "step_jobs_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "api_clients"
            referencedColumns: ["id"]
          },
        ]
      }
      telemetry_samples: {
        Row: {
          constraint_l2: number | null
          divergence_risk: number | null
          energy_drift_pct: number | null
          id: number
          ingested_at: string
          nan_count: number | null
          source: string | null
          t: number
          velocity_max: number | null
        }
        Insert: {
          constraint_l2?: number | null
          divergence_risk?: number | null
          energy_drift_pct?: number | null
          id?: number
          ingested_at?: string
          nan_count?: number | null
          source?: string | null
          t: number
          velocity_max?: number | null
        }
        Update: {
          constraint_l2?: number | null
          divergence_risk?: number | null
          energy_drift_pct?: number | null
          id?: number
          ingested_at?: string
          nan_count?: number | null
          source?: string | null
          t?: number
          velocity_max?: number | null
        }
        Relationships: []
      }
      user_feature_flags: {
        Row: {
          created_at: string
          enabled: boolean
          flag_key: string
          granted_by: string | null
          id: string
          user_id: string
        }
        Insert: {
          created_at?: string
          enabled: boolean
          flag_key: string
          granted_by?: string | null
          id?: string
          user_id: string
        }
        Update: {
          created_at?: string
          enabled?: boolean
          flag_key?: string
          granted_by?: string | null
          id?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_feature_flags_flag_key_fkey"
            columns: ["flag_key"]
            isOneToOne: false
            referencedRelation: "feature_flags"
            referencedColumns: ["key"]
          },
        ]
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
      is_admin: { Args: never; Returns: boolean }
    }
    Enums: {
      app_role: "user" | "enterprise" | "admin"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["user", "enterprise", "admin"],
    },
  },
} as const
