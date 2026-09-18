import { FolderOpen } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useRegisterSession } from "@/api/hooks/use-sessions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { t } from "@/i18n";
import { FsBrowserDialog } from "./FsBrowserDialog";

export function RegisterPathForm({ rid }: { rid: string }) {
  const navigate = useNavigate();
  const register = useRegisterSession(rid);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [browsing, setBrowsing] = useState(false);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = path.trim();
    if (!trimmed) return;
    register.mutate(
      { path: trimmed, name: name.trim() || undefined },
      {
        onSuccess: (session) => {
          toast.success(t("session.new.registered", { name: session.name }));
          navigate(`/regions/${rid}/sessions/${session.id}`);
        },
        onError: (err) => toast.error(errorDetail(err)),
      },
    );
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="space-y-1">
        <Label htmlFor="register-path">{t("session.new.pathLabel")}</Label>
        <div className="flex gap-2">
          <Input
            id="register-path"
            className="font-mono text-xs"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/Titan/dataset/…/000"
            required
          />
          <Button type="button" variant="outline" size="sm" onClick={() => setBrowsing(true)}>
            <FolderOpen className="mr-1 h-3.5 w-3.5" />
            {t("session.new.browse")}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">{t("session.new.pathHint")}</p>
      </div>
      <div className="space-y-1">
        <Label htmlFor="register-name">{t("session.new.nameLabel")}</Label>
        <Input id="register-name" value={name} onChange={(e) => setName(e.target.value)} />
      </div>
      <Button type="submit" size="sm" disabled={!path.trim() || register.isPending}>
        {t("session.new.submit")}
      </Button>
      <FsBrowserDialog
        open={browsing}
        onOpenChange={setBrowsing}
        onChoose={(p) => {
          setPath(p);
          setBrowsing(false);
        }}
      />
    </form>
  );
}
