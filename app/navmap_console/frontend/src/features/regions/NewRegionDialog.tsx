import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { z } from "zod";
import { useCreateRegion } from "@/api/hooks/use-regions";
import { errorDetail } from "@/api/client";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { t } from "@/i18n";

const positiveInt = z.number().int().positive();
const schema = z.object({
  name: z.string().trim().min(1).max(80),
  description: z.string().max(500),
  width: positiveInt,
  height: positiveInt,
  vprMethod: z.string().trim().min(1),
  vprBackbone: z.string().trim().min(1),
  vprDim: positiveInt,
});
type Values = z.infer<typeof schema>;

const defaults: Values = {
  name: "",
  description: "",
  width: 512,
  height: 288,
  vprMethod: "cosplace",
  vprBackbone: "ResNet18",
  vprDim: 256,
};

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NewRegionDialog({ open, onOpenChange }: Props) {
  const navigate = useNavigate();
  const create = useCreateRegion();
  const form = useForm<Values>({ resolver: zodResolver(schema), defaultValues: defaults });

  const submit = form.handleSubmit((v) => {
    create.mutate(
      {
        name: v.name,
        description: v.description,
        image_size: [v.width, v.height],
        vpr: { method: v.vprMethod, backbone: v.vprBackbone, dim: v.vprDim },
      },
      {
        onSuccess: (region) => {
          toast.success(t("regions.form.created", { name: region.name }));
          form.reset(defaults);
          onOpenChange(false);
          navigate(`/regions/${region.id}`);
        },
        onError: (err) => toast.error(errorDetail(err)),
      },
    );
  });

  // Number inputs hand back strings; convert on the way into the form state.
  const numberField = (field: { value: number; onChange: (v: number) => void }) => (
    <Input
      type="number"
      min={1}
      value={Number.isNaN(field.value) ? "" : field.value}
      onChange={(e) => field.onChange(e.target.valueAsNumber)}
    />
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t("regions.new")}</DialogTitle>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={submit} className="space-y-3">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{t("regions.form.name")}</FormLabel>
                  <FormControl>
                    <Input autoFocus placeholder="ucl_campus_aria" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{t("regions.form.description")}</FormLabel>
                  <FormControl>
                    <Textarea rows={2} {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div>
              <FormLabel>{t("regions.form.imageSize")}</FormLabel>
              <div className="mt-1 grid grid-cols-2 gap-2">
                <FormField
                  control={form.control}
                  name="width"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>{numberField(field)}</FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="height"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>{numberField(field)}</FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>
            <Accordion type="single" collapsible>
              <AccordionItem value="vpr" className="border-b-0">
                <AccordionTrigger className="py-2 text-xs">{t("regions.form.vpr")}</AccordionTrigger>
                <AccordionContent>
                  <FormDescription className="mb-2">{t("regions.form.vprHint")}</FormDescription>
                  <div className="grid grid-cols-3 gap-2">
                    <FormField
                      control={form.control}
                      name="vprMethod"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-xs">method</FormLabel>
                          <FormControl>
                            <Input {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="vprBackbone"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-xs">backbone</FormLabel>
                          <FormControl>
                            <Input {...field} />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="vprDim"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-xs">dim</FormLabel>
                          <FormControl>{numberField(field)}</FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
            <DialogFooter>
              <Button type="button" variant="outline" size="sm" onClick={() => onOpenChange(false)}>
                {t("common.cancel")}
              </Button>
              <Button type="submit" size="sm" disabled={create.isPending}>
                {t("common.create")}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
