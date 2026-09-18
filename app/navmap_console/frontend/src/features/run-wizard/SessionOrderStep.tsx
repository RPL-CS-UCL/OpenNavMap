import { DndContext, type DragEndEvent, PointerSensor, closestCenter, useSensor, useSensors } from "@dnd-kit/core";
import { SortableContext, arrayMove, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { GripVertical } from "lucide-react";
import type { Session } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { t } from "@/i18n";
import { seededShuffle } from "@/lib/seeded-shuffle";

interface Props {
  sessions: Session[];
  order: string[];
  seed: number | null;
  onOrder: (order: string[]) => void;
  onSeed: (seed: number | null) => void;
}

function Row({ session, index }: { session: Session; index: number }) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: session.id });
  return (
    <li ref={setNodeRef} style={{ transform: CSS.Transform.toString(transform), transition }} className="flex items-center gap-2 border-b px-2 py-1 text-xs">
      <button type="button" {...attributes} {...listeners} aria-label={`drag ${session.name}`} className="cursor-grab text-muted-foreground">
        <GripVertical className="h-3.5 w-3.5" />
      </button>
      <span className="w-6 font-mono tabular-nums text-muted-foreground">{index}</span>
      <span className="truncate">{session.name}</span>
      <span className="ml-auto font-mono tabular-nums text-muted-foreground">{session.num_frames}</span>
    </li>
  );
}

export function SessionOrderStep({ sessions, order, seed, onOrder, onSeed }: Props) {
  const sensors = useSensors(useSensor(PointerSensor));
  const byId = new Map(sessions.map((s) => [s.id, s]));
  const toggle = (id: string, on: boolean) => onOrder(on ? [...order, id] : order.filter((x) => x !== id));
  const onDragEnd = (e: DragEndEvent) => {
    if (!e.over || e.active.id === e.over.id) return;
    onOrder(arrayMove(order, order.indexOf(String(e.active.id)), order.indexOf(String(e.over.id))));
  };
  const byName = () => onOrder([...order].sort((a, b) => (byId.get(a)?.name ?? "").localeCompare(byId.get(b)?.name ?? "")));
  const shuffle = () => {
    const s = seed ?? Math.floor(Math.random() * 1e9);
    onSeed(s);
    onOrder(seededShuffle(order, s));
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <section>
        <h3 className="mb-1 text-sm font-semibold">{t("wizard.sessions.available")}</h3>
        <ul className="border">
          {sessions.map((s) => {
            const invalid = !!s.validation && !s.validation.ok;
            return (
              <li key={s.id} className="flex items-center gap-2 border-b px-2 py-1 text-xs">
                <Checkbox id={`pick-${s.id}`} checked={order.includes(s.id)} disabled={invalid} onCheckedChange={(v) => toggle(s.id, v === true)} aria-label={s.name} />
                <label htmlFor={`pick-${s.id}`} className="truncate">
                  {s.name}
                </label>
                {invalid && <span className="ml-auto text-destructive">{t("wizard.sessions.invalid")}</span>}
              </li>
            );
          })}
        </ul>
      </section>
      <section>
        <div className="mb-1 flex items-center gap-1">
          <h3 className="text-sm font-semibold">{t("wizard.sessions.order")}</h3>
          <Button size="sm" variant="ghost" className="ml-auto h-6 px-2 text-xs" onClick={byName}>
            {t("wizard.order.byName")}
          </Button>
          <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={() => onOrder([...order].reverse())}>
            {t("wizard.order.reverse")}
          </Button>
          <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={shuffle}>
            {t("wizard.order.shuffle")}
          </Button>
          <Input
            type="number"
            aria-label={t("wizard.order.seed")}
            placeholder={t("wizard.order.seed")}
            className="h-6 w-24 font-mono text-xs"
            value={seed ?? ""}
            onChange={(e) => onSeed(e.target.value === "" ? null : Number(e.target.value))}
          />
        </div>
        {order.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("wizard.sessions.none")}</p>
        ) : (
          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
            <SortableContext items={order} strategy={verticalListSortingStrategy}>
              <ul className="border">
                {order.map((id, i) => {
                  const s = byId.get(id);
                  return s ? <Row key={id} session={s} index={i} /> : null;
                })}
              </ul>
            </SortableContext>
          </DndContext>
        )}
      </section>
    </div>
  );
}
