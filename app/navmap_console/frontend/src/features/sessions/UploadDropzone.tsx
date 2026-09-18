import { FileArchive, Upload } from "lucide-react";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { errorDetail } from "@/api/client";
import { useUploadSession } from "@/api/hooks/use-sessions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { t } from "@/i18n";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

export function UploadDropzone({ rid }: { rid: string }) {
  const navigate = useNavigate();
  const upload = useUploadSession(rid);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted[0]) setFile(accepted[0]);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    multiple: false,
    accept: { "application/zip": [".zip"], "application/x-7z-compressed": [".7z"] },
  });

  const submit = () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    if (name.trim()) form.append("name", name.trim());
    upload.mutate(form, {
      onSuccess: (session) => {
        toast.success(t("session.new.uploaded", { name: session.name }));
        navigate(`/regions/${rid}/sessions/${session.id}`);
      },
      onError: (err) => toast.error(errorDetail(err)),
    });
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">{t("session.new.uploadHint")}</p>
      <div
        {...getRootProps()}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-sm border border-dashed p-8 text-sm text-muted-foreground",
          isDragActive && "border-primary bg-accent",
        )}
      >
        <input {...getInputProps()} aria-label={t("session.new.uploadPick")} />
        {file ? (
          <>
            <FileArchive className="h-5 w-5" />
            <span className="font-mono text-xs text-foreground">
              {file.name} · {formatBytes(file.size)}
            </span>
          </>
        ) : (
          <>
            <Upload className="h-5 w-5" />
            <span>{t("session.new.uploadPick")}</span>
          </>
        )}
      </div>
      <div className="space-y-1">
        <Label htmlFor="upload-name">{t("session.new.nameLabel")}</Label>
        <Input id="upload-name" value={name} onChange={(e) => setName(e.target.value)} />
      </div>
      {upload.isPending && file && (
        <div className="space-y-1">
          <div className="text-xs">{t("session.new.uploading", { name: file.name, size: formatBytes(file.size) })}</div>
          <div className="h-1 w-full overflow-hidden rounded-sm bg-muted">
            <div className="h-full w-1/3 animate-pulse bg-primary" />
          </div>
        </div>
      )}
      <Button size="sm" onClick={submit} disabled={!file || upload.isPending}>
        {t("session.new.upload")}
      </Button>
    </div>
  );
}
