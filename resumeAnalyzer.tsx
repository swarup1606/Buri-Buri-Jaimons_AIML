import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { toast } from "@/components/ui/use-toast";
import { Loader2 } from 'lucide-react';

interface ResumeResult {
  names: string[];
  emails: string[];
  similarity: number;
  selected: boolean;
  text: string;
}

const API_BASE_URL = 'http://localhost:5000/api';

const PREDEFINED_MESSAGES = {
  accept: `Dear Candidate,

We are pleased to inform you that your application has been selected for the position. Your qualifications and experience align well with our requirements.

Next Steps:
1. Please confirm your availability for an interview
2. Prepare any questions you may have about the role
3. We will contact you shortly to schedule the interview

Best regards,
HR Team`,

  reject: `Dear Candidate,

Thank you for your interest in the position. After careful review of your application, we regret to inform you that we have decided to move forward with other candidates whose qualifications more closely match our current needs.

We appreciate your time and interest in our company. We encourage you to apply for future positions that match your skills and experience.

Best regards,
HR Team`
};

const ResumeAnalyzer: React.FC = () => {
  const [jobDescription, setJobDescription] = useState<string>('');
  const [resumeFiles, setResumeFiles] = useState<File[]>([]);
  const [threshold, setThreshold] = useState<number>(50);
  const [results, setResults] = useState<ResumeResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedResumes, setSelectedResumes] = useState<number[]>([]);
  const [customMessage, setCustomMessage] = useState<string>('');
  const [generatedMessage, setGeneratedMessage] = useState<string>('');
  const [darkMode, setDarkMode] = useState<boolean>(false);

  // Add effect to log results changes
  useEffect(() => {
    console.log('Results state updated:', results);
  }, [results]);

  // Add effect to update selections when threshold changes
  useEffect(() => {
    if (results.length > 0) {
      const selectedIndices = results
        .map((result, index) => result.similarity >= threshold ? index : -1)
        .filter(index => index !== -1);
      setSelectedResumes(selectedIndices);
    }
  }, [threshold, results]);

  // Initialize dark mode from localStorage
  useEffect(() => {
    const dark = localStorage.getItem("dark-mode") === "enabled";
    setDarkMode(dark);
    if (dark) {
      document.documentElement.classList.add("dark");
      document.body.classList.add("dark-mode");
    }
  }, []);

  // Toggle dark mode
  const toggleDarkMode = (): void => {
    const isDark = !darkMode;
    setDarkMode(isDark);
    
    if (isDark) {
      document.documentElement.classList.add("dark");
      document.body.classList.add("dark-mode");
      localStorage.setItem("dark-mode", "enabled");
    } else {
      document.documentElement.classList.remove("dark");
      document.body.classList.remove("dark-mode");
      localStorage.setItem("dark-mode", "disabled");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files) {
      setResumeFiles(Array.from(e.target.files));
    }
  };

  const handleAnalyze = async () => {
    if (!jobDescription.trim()) {
      toast({
        title: "Error",
        description: "Please enter a job description",
        variant: "destructive",
      });
      return;
    }

    if (resumeFiles.length === 0) {
      toast({
        title: "Error",
        description: "Please upload at least one resume",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('job_description', jobDescription);
      resumeFiles.forEach((file) => {
        formData.append('resume_files', file);
      });
      formData.append('threshold', threshold.toString());

      console.log('Sending request with:', {
        jobDescription: jobDescription.substring(0, 100) + '...',
        numFiles: resumeFiles.length,
        threshold
      });

      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received data from backend:', data);
      
      if (data.success) {
        if (!data.results || data.results.length === 0) {
          toast({
            title: "Warning",
            description: "No matching results found. Try adjusting the similarity threshold or job description.",
            variant: "destructive",
          });
          return;
        }
        setResults(data.results);
        // Automatically select candidates based on threshold
        const selectedIndices = data.results
          .map((result, index) => result.similarity >= threshold ? index : -1)
          .filter(index => index !== -1);
        setSelectedResumes(selectedIndices);
        toast({
          title: "Success",
          description: "Resumes analyzed successfully!",
        });
      } else {
        throw new Error(data.error || 'Failed to analyze resumes');
      }
    } catch (error) {
      console.error('Error analyzing resumes:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to analyze resumes",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMessage = async (action: 'accept' | 'reject'): Promise<void> => {
    try {
      if (results.length === 0) {
        toast({
          title: "Error",
          description: "Please analyze resumes first before generating messages.",
          variant: "destructive",
        });
        return;
      }

      try {
        const formData = new FormData();
        formData.append('action', action);
        formData.append('job_description', jobDescription);
        
        // Add selected candidates' information
        const selectedCandidates = selectedResumes.map(index => ({
          name: results[index].names[0] || 'Name not found',
          email: results[index].emails[0] || 'Email not found',
          similarity: results[index].similarity
        }));
        formData.append('candidates', JSON.stringify(selectedCandidates));

        const response = await fetch(`${API_BASE_URL}/generate-message`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
          setGeneratedMessage(data.generated_message);
          setCustomMessage(data.generated_message);
          toast({
            title: "Success",
            description: `${action === 'accept' ? 'Acceptance' : 'Rejection'} message generated successfully!`,
          });
        } else {
          throw new Error(data.error || 'Failed to generate message');
        }
      } catch (error) {
        console.error('Error generating message:', error);
        toast({
          title: "Error",
          description: "Failed to generate message. Please try again.",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate message",
        variant: "destructive",
      });
    }
  };

  const handleProcessAction = async (action: 'accept' | 'reject'): Promise<void> => {
    if (action === 'accept' && selectedResumes.length === 0) {
      toast({
        title: "Error",
        description: "Please select at least one resume to accept.",
        variant: "destructive",
      });
      return;
    }

    try {
      let messageToUse = customMessage;
      
      // If no custom message exists, try to generate one
      if (!messageToUse) {
        try {
          const messageFormData = new FormData();
          messageFormData.append('action', action);
          
          const messageResponse = await fetch(`${API_BASE_URL}/generate-message`, {
            method: 'POST',
            body: messageFormData,
          });

          if (!messageResponse.ok) {
            throw new Error(`Failed to generate message: ${messageResponse.status}`);
          }

          const messageData = await messageResponse.json();
          if (messageData.success) {
            messageToUse = messageData.generated_message;
          } else {
            throw new Error(messageData.error || 'Failed to generate message');
          }
        } catch (error) {
          // If AI generation fails, use predefined message without notification
          messageToUse = PREDEFINED_MESSAGES[action];
        }
      }

      // Process the action with the message
      const formData = new FormData();
      formData.append('action', action);
      formData.append('custom_message', messageToUse);
      selectedResumes.forEach(index => {
        formData.append('selected_resume', index.toString());
      });

      const response = await fetch(`${API_BASE_URL}/process-action`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.success) {
        setCustomMessage(messageToUse);
        setGeneratedMessage(messageToUse);
        toast({
          title: "Success",
          description: "Action processed successfully!",
        });
        console.log(data.msg);
      } else {
        throw new Error(data.msg || 'Failed to process action');
      }
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process action",
        variant: "destructive",
      });
    }
  };

  const handleDownloadCSV = async (): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/download-csv`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ranked_resumes.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast({
        title: "Success",
        description: "CSV file downloaded successfully!",
      });
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "Error",
        description: "Failed to download CSV file",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 dark:bg-gray-900">
      <div className="flex justify-center items-center mb-8">
        <h1 className="text-4xl font-bold dark:text-white">Multiple Resume Analyzer</h1>
      </div>

      <div className="flex justify-end mb-4">
        <label className="flex items-center space-x-2 dark:text-white">
          <input
            type="checkbox"
            checked={darkMode}
            onChange={toggleDarkMode}
            className="form-checkbox h-4 w-4 text-blue-600"
          />
          <span>Dark Mode</span>
        </label>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-6">
          <Card className="p-6 dark:bg-gray-800 dark:border-gray-700">
            <h2 className="text-xl font-semibold mb-4 dark:text-white">Job Description</h2>
            <Textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Enter the job description..."
              className="min-h-[200px] dark:bg-gray-700 dark:text-white dark:border-gray-600"
            />
          </Card>

          <Card className="p-6 dark:bg-gray-800 dark:border-gray-700">
            <h2 className="text-xl font-semibold mb-4 dark:text-white">Upload Resumes</h2>
            <Input
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileChange}
              className="mb-4 dark:bg-gray-700 dark:text-white dark:border-gray-600"
            />
            <div className="space-y-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Selected files:</p>
              {resumeFiles.map((file, index) => (
                <p key={index} className="text-sm dark:text-white">{file.name}</p>
              ))}
            </div>
          </Card>

          <Card className="p-6 dark:bg-gray-800 dark:border-gray-700">
            <h2 className="text-xl font-semibold mb-4 dark:text-white">Similarity Threshold</h2>
            <div className="space-y-4">
              <Slider
                value={[threshold]}
                onValueChange={(value) => setThreshold(value[0])}
                min={0}
                max={100}
                step={1}
                className="dark:bg-gray-700"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400">Threshold: {threshold}%</p>
            </div>
          </Card>

          <Button
            onClick={handleAnalyze}
            disabled={loading || !jobDescription || resumeFiles.length === 0}
            className="w-full dark:bg-blue-600 dark:hover:bg-blue-700"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze Resumes'
            )}
          </Button>
        </div>

        <div className="space-y-6">
          {results.length > 0 && (
            <Card className="p-6 dark:bg-gray-800 dark:border-gray-700">
              <h2 className="text-xl font-semibold mb-4 dark:text-white">Results</h2>
              <div className="space-y-4">
                {results.map((result, index) => (
                  <div key={index} className="border rounded-lg p-4 dark:border-gray-700 dark:bg-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <p className="font-medium dark:text-white">
                          {result.names[0] || 'Name not found'}
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {result.emails[0] || 'Email not found'}
                        </p>
                      </div>
                      <div className="flex items-center space-x-4">
                        <p className="text-sm dark:text-white">
                          ATS Similarity: {result.similarity.toFixed(2)}%
                        </p>
                        <Checkbox
                          checked={selectedResumes.includes(index)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedResumes([...selectedResumes, index]);
                            } else {
                              setSelectedResumes(selectedResumes.filter(i => i !== index));
                            }
                          }}
                          className="dark:border-gray-600"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 space-y-4">
                <div className="flex space-x-4">
                  <Button
                    onClick={() => handleGenerateMessage('accept')}
                    variant="outline"
                    className="dark:border-gray-600 dark:text-white dark:hover:bg-gray-700"
                  >
                    Generate Acceptance Message
                  </Button>
                  <Button
                    onClick={() => handleGenerateMessage('reject')}
                    variant="outline"
                    className="dark:border-gray-600 dark:text-white dark:hover:bg-gray-700"
                  >
                    Generate Rejection Message
                  </Button>
                </div>

                <Textarea
                  value={customMessage}
                  onChange={(e) => setCustomMessage(e.target.value)}
                  placeholder="Customize the message..."
                  className="min-h-[300px] dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />

                <div className="flex space-x-4">
                  <Button
                    onClick={() => handleProcessAction('accept')}
                    disabled={selectedResumes.length === 0}
                    className="dark:bg-blue-600 dark:hover:bg-blue-700"
                  >
                    Accept Selected
                  </Button>
                  <Button
                    onClick={() => handleProcessAction('reject')}
                    variant="destructive"
                    className="dark:bg-red-600 dark:hover:bg-red-700"
                  >
                    Reject Others
                  </Button>
                  <Button
                    onClick={handleDownloadCSV}
                    variant="outline"
                    className="dark:border-gray-600 dark:text-white dark:hover:bg-gray-700"
                  >
                    Download CSV
                  </Button>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResumeAnalyzer; 