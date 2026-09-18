import { useCallback, useEffect, useRef, useState } from "react";
import { fetchJobLog } from "@/api/hooks/use-jobs";
import type { Job } from "@/api/types";
import { useTopic } from "@/ws/use-socket";

const RING = 20_000;

interface Snapshot {
  job: Job;
  lines: string[];
  next_seq: number;
}
interface Chunk {
  id: string;
  first_seq: number;
  lines: string[];
}

/** Log lines for one job: REST for the initial page and gap-fill, WebSocket for the live tail. */
export function useJobLog(jobId: string | null): { lines: string[] } {
  const [lines, setLines] = useState<string[]>([]);
  const nextSeq = useRef(0);
  const filling = useRef(false);

  const append = useCallback(
    (incoming: string[], firstSeq: number) => {
      if (firstSeq > nextSeq.current) {
        // gap between what we have and what arrived: fetch the missing range first
        if (!jobId || filling.current) return;
        filling.current = true;
        fetchJobLog(jobId, nextSeq.current)
          .then((chunk) => {
            nextSeq.current = chunk.next;
            setLines((prev) => [...prev, ...chunk.lines].slice(-RING));
          })
          .finally(() => {
            filling.current = false;
          });
        return;
      }
      const fresh = incoming.slice(nextSeq.current - firstSeq);
      if (fresh.length === 0) return;
      nextSeq.current = firstSeq + incoming.length;
      setLines((prev) => [...prev, ...fresh].slice(-RING));
    },
    [jobId],
  );

  useEffect(() => {
    setLines([]);
    nextSeq.current = 0;
    if (!jobId) return;
    let alive = true;
    fetchJobLog(jobId, 0).then((chunk) => {
      if (!alive) return;
      nextSeq.current = chunk.next;
      setLines(chunk.lines.slice(-RING));
    });
    return () => {
      alive = false;
    };
  }, [jobId]);

  useTopic(jobId ? `job:${jobId}` : null, (m) => {
    if (m.type === "job.snapshot") {
      const snap = m.data as Snapshot;
      nextSeq.current = snap.next_seq;
      setLines(snap.lines.slice(-RING));
    } else if (m.type === "job.log") {
      const chunk = m.data as Chunk;
      append(chunk.lines, chunk.first_seq);
    }
  });

  return { lines };
}
