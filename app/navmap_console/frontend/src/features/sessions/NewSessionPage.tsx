import { useParams } from "react-router-dom";
import { PageHeader } from "@/components/common/PageHeader";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { t } from "@/i18n";
import { RegisterPathForm } from "./RegisterPathForm";
import { UploadDropzone } from "./UploadDropzone";

export function NewSessionPage() {
  const { rid = "" } = useParams();
  return (
    <div className="max-w-2xl">
      <PageHeader backTo={`/regions/${rid}`} title={t("session.new.title")} />
      <Tabs defaultValue="register">
        <TabsList>
          <TabsTrigger value="register">{t("session.new.register")}</TabsTrigger>
          <TabsTrigger value="upload">{t("session.new.upload")}</TabsTrigger>
        </TabsList>
        <TabsContent value="register" className="pt-3">
          <RegisterPathForm rid={rid} />
        </TabsContent>
        <TabsContent value="upload" className="pt-3">
          <UploadDropzone rid={rid} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
