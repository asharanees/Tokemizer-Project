import { useState, useEffect } from "react";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
    Zap,
    ShieldCheck,
    BarChart3,
    Map as MapIcon,
    ChevronRight,
    ChevronLeft
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const steps = [
    {
        title: "Welcome to Tokemizer",
        description: "The intelligent prompt optimizer that saves tokens without losing meaning. Let's get you started!",
        icon: <Zap className="h-12 w-12 text-primary" />,
        color: "bg-primary/10"
    },
    {
        title: "Smart Optimization",
        description: "Our multi-pass engine uses techniques like stop-word removal, entity canonicalization, and telegram-style grammar reduction.",
        icon: <ShieldCheck className="h-12 w-12 text-green-500" />,
        color: "bg-green-500/10"
    },
    {
        title: "Canonical Mappings",
        description: "Define custom rules to normalize frequent tokens (e.g., 'Acme Corporation' -> 'Acme'). A huge way to save tokens globally.",
        icon: <MapIcon className="h-12 w-12 text-blue-500" />,
        color: "bg-blue-500/10"
    },
    {
        title: "Monitor & Save",
        description: "Track your optimization trends and total savings in real-time on your dashboard.",
        icon: <BarChart3 className="h-12 w-12 text-purple-500" />,
        color: "bg-purple-500/10"
    }
];

export function Onboarding() {
    const { user } = useAuth();
    const [open, setOpen] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);

    useEffect(() => {
        if (!user) return;

        const hasSeenOnboarding = localStorage.getItem(`onboarding_seen_${user.id}`);
        if (!hasSeenOnboarding) {
            setOpen(true);
        }
    }, [user]);

    const handleNext = () => {
        if (currentStep < steps.length - 1) {
            setCurrentStep(currentStep + 1);
        } else {
            finishOnboarding();
        }
    };

    const handleBack = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
        }
    };

    const finishOnboarding = () => {
        if (user) {
            localStorage.setItem(`onboarding_seen_${user.id}`, "true");
        }
        setOpen(false);
    };

    const step = steps[currentStep];

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent className="sm:max-w-[425px]">
                <div className="flex flex-col items-center text-center py-6">
                    <div className={`p-6 rounded-full ${step.color} mb-6 transition-colors duration-500`}>
                        {step.icon}
                    </div>
                    <DialogHeader>
                        <DialogTitle className="text-2xl font-bold">{step.title}</DialogTitle>
                        <DialogDescription className="text-base mt-2">
                            {step.description}
                        </DialogDescription>
                    </DialogHeader>
                </div>

                <div className="flex justify-center gap-1.5 mb-4">
                    {steps.map((_, i) => (
                        <div
                            key={i}
                            className={`h-1.5 rounded-full transition-all duration-300 ${i === currentStep ? "w-6 bg-primary" : "w-1.5 bg-muted"}`}
                        />
                    ))}
                </div>

                <DialogFooter className="flex-row sm:justify-between items-center">
                    <Button
                        variant="ghost"
                        onClick={handleBack}
                        disabled={currentStep === 0}
                        className="px-2"
                        tooltip={currentStep === 0 ? "You're at the first step" : "Go back to the previous step"}
                    >
                        <ChevronLeft className="h-4 w-4 mr-1" /> Back
                    </Button>

                    <Button onClick={handleNext} className="shadow-neon-sm" tooltip={currentStep === steps.length - 1 ? "Complete onboarding and start using the app" : "Continue to the next step"}>
                        {currentStep === steps.length - 1 ? "Get Started" : "Next"}
                        {currentStep < steps.length - 1 && <ChevronRight className="h-4 w-4 ml-1" />}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
