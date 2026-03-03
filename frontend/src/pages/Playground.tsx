import { Layout } from "@/components/layout/Layout";
import { OptimizerPlayground } from "@/components/optimizer/OptimizerPlayground";
import { NavigationBreadcrumb } from "@/components/layout/NavigationBreadcrumb";

export default function Playground() {
  return (
    <Layout>
      <div className="space-y-2 lg:space-y-4 h-full flex flex-col">
        <NavigationBreadcrumb />
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-display text-glow">Optimizer Playground</h1>
          <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">Test optimization algorithms in real-time with semantic preservation checks.</p>
        </div>

        <div className="flex-1 min-h-0">
          <OptimizerPlayground />
        </div>
      </div>
    </Layout>
  );
}
