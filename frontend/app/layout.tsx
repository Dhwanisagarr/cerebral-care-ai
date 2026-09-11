import type { Metadata } from "next";
import "./globals.css";
import MedicalDisclaimerBanner from "@/components/MedicalDisclaimerBanner";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "Assistive AI for Speech & Pain Detection | Cerebral Palsy Assistive Tech",
  description: "AI-assisted speech clarity assessment and facial image pain indicator analysis for individuals with cerebral palsy.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
        <MedicalDisclaimerBanner />
        <Navbar />
        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
