import { useCallback, useEffect, useRef, useState } from "react";

/** Width of an element, tracked with a ResizeObserver. */
export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/** View state kept in the URL query, so a view can be bookmarked or shared. */
export function useUrlState<K extends string>(
  defaults: Record<K, string>,
): [Record<K, string>, (patch: Partial<Record<K, string>>) => void] {
  const read = useCallback(() => {
    const query = new URLSearchParams(window.location.search);
    const state = { ...defaults };
    for (const key of Object.keys(defaults) as K[]) {
      const value = query.get(key);
      if (value !== null) state[key] = value;
    }
    return state;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [state, setState] = useState(read);
  useEffect(() => {
    const onPop = () => setState(read());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [read]);
  const update = useCallback((patch: Partial<Record<K, string>>) => {
    setState((previous) => {
      const next = { ...previous, ...patch };
      const query = new URLSearchParams();
      for (const [key, value] of Object.entries(next) as [K, string][]) {
        if (value !== "") query.set(key, value);
      }
      window.history.pushState(null, "", `?${query}`);
      return next;
    });
  }, []);
  return [state, update];
}

/** Fetch on key change; the previous data stays (marked stale) while the next loads. */
export function useFetch<T>(
  key: string | null,
  load: () => Promise<T>,
): { data: T | null; error: string | null; stale: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const loader = useRef(load);
  loader.current = load;
  useEffect(() => {
    if (key === null) {
      setData(null);
      setError(null);
      setLoadedKey(null);
      return;
    }
    let live = true;
    loader
      .current()
      .then((value) => {
        if (!live) return;
        setData(value);
        setError(null);
        setLoadedKey(key);
      })
      .catch((reason: Error) => {
        if (!live) return;
        setError(reason.message);
        setLoadedKey(key);
      });
    return () => {
      live = false;
    };
  }, [key]);
  return { data, error, stale: key !== loadedKey };
}
