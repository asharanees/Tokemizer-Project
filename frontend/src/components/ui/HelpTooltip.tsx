import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

export function HelpTooltip({ content }: { content: string }) {
    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help transition-colors" />
                </TooltipTrigger>
                <TooltipContent className="max-w-[200px] text-xs">
                    <p>{content}</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}
