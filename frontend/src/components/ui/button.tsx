import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 cursor-pointer" +
" hover-elevate active-elevate-2",
  {
    variants: {
      variant: {
        default:
           // Enhanced with hover effects and proper focus states
           "bg-primary text-primary-foreground border border-primary-border hover:bg-primary/90 hover:shadow-md focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm border-destructive-border hover:bg-destructive/90 hover:shadow-md focus-visible:ring-2 focus-visible:ring-destructive focus-visible:ring-offset-2",
        outline:
          // Enhanced with hover background and better focus states
          "border [border-color:var(--button-outline)] shadow-xs hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 active:shadow-none",
        secondary:
          // Enhanced with hover effects and proper focus states
          "border bg-secondary text-secondary-foreground border border-secondary-border hover:bg-secondary/80 hover:shadow-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        // Enhanced ghost variant with hover background
        ghost: "border border-transparent hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        link: "text-primary underline-offset-4 hover:underline focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2",
      },
      size: {
        // @replit changed sizes
        default: "min-h-9 px-4 py-2",
        sm: "min-h-8 rounded-md px-3 text-xs",
        lg: "min-h-10 rounded-md px-8",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  tooltip?: string | React.ReactNode
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, tooltip, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    const buttonElement = (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )

    if (!tooltip) {
      return buttonElement
    }

    return (
      <Tooltip>
        <TooltipTrigger asChild>{buttonElement}</TooltipTrigger>
        <TooltipContent>
          {typeof tooltip === 'string' ? tooltip : tooltip}
        </TooltipContent>
      </Tooltip>
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
