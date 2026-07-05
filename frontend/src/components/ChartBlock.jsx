import React from 'react'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    LineChart, Line, PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00C49F', '#FFBB28', '#FF8042']

export default function ChartBlock({ config }) {
    if (!config || !config.data || config.data.length === 0) return null

    const { chartType, xAxisKey, yAxisKey, description, data } = config

    const renderChart = () => {
        switch (chartType?.toLowerCase()) {
            case 'line':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey={xAxisKey} stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: '#222', borderColor: '#444' }} />
                            <Legend />
                            <Line type="monotone" dataKey={yAxisKey} stroke="#8884d8" activeDot={{ r: 8 }} />
                        </LineChart>
                    </ResponsiveContainer>
                )
            case 'pie':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                outerRadius={100}
                                fill="#8884d8"
                                dataKey={yAxisKey}
                                nameKey={xAxisKey}
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#222', borderColor: '#444' }} />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                )
            case 'bar':
            default:
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey={xAxisKey} stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: '#222', borderColor: '#444', color: '#fff' }} cursor={{fill: '#333'}} />
                            <Legend />
                            <Bar dataKey={yAxisKey} fill="#8884d8" />
                        </BarChart>
                    </ResponsiveContainer>
                )
        }
    }

    return (
        <div className="chart-block" style={{ margin: '20px 0', padding: '16px', backgroundColor: 'var(--bg-lighter)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <h4 style={{ margin: '0 0 16px 0', fontSize: '0.95rem', color: 'var(--text-light)' }}>
                📊 {description || 'Data Visualization'}
            </h4>
            {renderChart()}
        </div>
    )
}
