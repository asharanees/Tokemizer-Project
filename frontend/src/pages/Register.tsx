import { useEffect } from "react";
import { useLocation } from "wouter";
import { useSeo } from "@/lib/seo";

export default function Register() {
    useSeo({
        title: "Register",
        path: "/register",
        robots: "noindex,follow",
    });

    const [, setLocation] = useLocation();

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        params.set("action", "register");
        setLocation(`/login?${params.toString()}`);
    }, [setLocation]);

    return null;
}
