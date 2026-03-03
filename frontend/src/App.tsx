import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/Dashboard";
import Playground from "@/pages/Playground";
import Settings from "@/pages/Settings";
import Analytics from "@/pages/Analytics";
import History from "@/pages/History";
import BatchJobs from "@/pages/BatchJobs";
import CanonicalMappings from "@/pages/CanonicalMappings";
import Profile from "@/pages/Profile";
import ApiKeys from "@/pages/ApiKeys";
import Subscription from "@/pages/Subscription";
import AdminUsers from "@/pages/admin/Users";
import AdminPlans from "@/pages/admin/Plans";
import AdminSettings from "@/pages/admin/Settings";
import AdminOOTB from "@/pages/admin/OOTBMappings";
import AdminModels from "@/pages/admin/Models";
import TenantHealth from "@/pages/admin/TenantHealth";

import { AuthProvider } from "@/contexts/AuthContext";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import About from "@/pages/About";
import Plan from "@/pages/Plan";
import Contact from "@/pages/Contact";
import FAQ from "@/pages/FAQ";
import ProtectedRoute from "@/components/ProtectedRoute";
import AdminRoute from "@/components/AdminRoute";

function Router() {
  return (
    <Switch>
      <Route path="/login" component={Login} />
      <Route path="/register" component={Register} />
      <Route path="/about" component={About} />
      <Route path="/plans" component={Plan} />
      <Route path="/contact" component={Contact} />
      <Route path="/faq" component={FAQ} />

      {/* Protected Routes */}
      <ProtectedRoute path="/" component={Dashboard} />
      <ProtectedRoute path="/playground" component={Playground} />
      <ProtectedRoute path="/batch" component={BatchJobs} />
      <ProtectedRoute path="/analytics" component={Analytics} />
      <ProtectedRoute path="/history" component={History} />
      <ProtectedRoute path="/settings" component={Settings} />
      <ProtectedRoute path="/canonical" component={CanonicalMappings} />
      <ProtectedRoute path="/profile" component={Profile} />
      <ProtectedRoute path="/keys" component={ApiKeys} />
      <ProtectedRoute path="/subscription" component={Subscription} />

      {/* Admin Routes */}
      <AdminRoute path="/admin/users" component={AdminUsers} />
      <AdminRoute path="/admin/plans" component={AdminPlans} />
      <AdminRoute path="/admin/settings" component={AdminSettings} />
      <AdminRoute path="/admin/ootb" component={AdminOOTB} />
      <AdminRoute path="/admin/models" component={AdminModels} />
      <AdminRoute path="/admin/tenant-health" component={TenantHealth} />

      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
